import heapq
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from crazyflie_interfaces.msg import Position
from crazyflie_interfaces.srv import Takeoff
from nav_msgs.msg import OccupancyGrid

class DroneTracker(Node):
    def __init__(self):
        super().__init__('drone_tracker_node')

        self.cruise_height = 1.0
        self.max_speed = 2.0
        self.control_period = 0.1
        self.max_step = self.max_speed * self.control_period

        self.safety_margin = 0.35
        self.grid_res = 0.4
        self.replan_period = 1.0

        self.obstacles = []
        self.have_obstacles = False
        self.path = []
        self.path_index = 0

        self.create_subscription(Point, '/AGV/pose', self.truck_callback, 10)
        self.create_subscription(PoseStamped, '/cf_1/pose', self.drone_callback, 10)
        
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        
        self.publisher = self.create_publisher(Position, '/cf_1/cmd_position', 10)

        self.truck_x = 0.0
        self.truck_y = 0.0
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = 0.0
        self.have_truck = False
        self.have_drone = False

        self.takeoff_client = self.create_client(Takeoff, '/cf_1/takeoff')
        while not self.takeoff_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the flight controller...')

        req = Takeoff.Request()
        req.height = self.cruise_height
        req.duration.sec = 2
        req.duration.nanosec = 0
        self.takeoff_client.call_async(req)
        self.get_logger().info('Takeoff commanded! Waiting 3 seconds...')

        self.is_tracking = False
        self.start_timer = self.create_timer(3.0, self.start_tracking)
        self.timer = self.create_timer(self.control_period, self.tracking_loop)
        self.replan_timer = self.create_timer(self.replan_period, self.replan)

    def start_tracking(self):
        if not (self.have_truck and self.have_drone):
            self.get_logger().warn('No pose data yet from /AGV/pose or /cf_1/pose -- check those topics.')
        self.is_tracking = True
        self.start_timer.cancel()
        self.get_logger().info('Tracker engaged.')

    def truck_callback(self, msg):
        self.truck_x = msg.x
        self.truck_y = msg.y
        self.have_truck = True

    def drone_callback(self, msg):
        self.drone_x = msg.pose.position.x
        self.drone_y = msg.pose.position.y
        self.drone_z = msg.pose.position.z
        self.have_drone = True

    def map_callback(self, msg):
        if self.have_obstacles:
            return
        
        res = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        width = msg.info.width
        
        obstacles = []
        
        for i, val in enumerate(msg.data):
            if val > 50: 
                grid_y = i // width
                grid_x = i % width
                
                real_x = origin_x + (grid_x * res) + (res / 2.0)
                real_y = origin_y + (grid_y * res) + (res / 2.0)
                
                obstacles.append((real_x, real_y, (res / 2.0) + self.safety_margin))
                
        if obstacles:
            self.obstacles = obstacles
            self.have_obstacles = True
            self.get_logger().info(f'Successfully locked onto {len(obstacles)} obstacles from the 2D map!')

    def is_blocked(self, x, y):
        for ox, oy, r in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r * r:
                return True
        return False

    def astar(self, start, goal):
        res = self.grid_res
        pad = 2.0
        min_x = min(start[0], goal[0]) - pad
        max_x = max(start[0], goal[0]) + pad
        min_y = min(start[1], goal[1]) - pad
        max_y = max(start[1], goal[1]) + pad

        def to_cell(x, y):
            return (round((x - min_x) / res), round((y - min_y) / res))

        def to_world(c):
            return (min_x + c[0] * res, min_y + c[1] * res)

        start_cell = to_cell(*start)
        goal_cell = to_cell(*goal)
        nx = int((max_x - min_x) / res) + 2
        ny = int((max_y - min_y) / res) + 2

        def blocked(cell):
            wx, wy = to_world(cell)
            return self.is_blocked(wx, wy)

        if blocked(start_cell):
            return None

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        open_set = [(0.0, start_cell)]
        came_from = {}
        g_score = {start_cell: 0.0}
        visited = set()

        def heuristic(c):
            return math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])

        expanded = 0
        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            expanded += 1
            if current == goal_cell:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [to_world(c) for c in path]
            if expanded > 6000:
                break 
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] <= nx and 0 <= nxt[1] <= ny):
                    continue
                if blocked(nxt):
                    continue
                step_cost = math.hypot(dx, dy)
                tentative = g_score[current] + step_cost
                if tentative < g_score.get(nxt, float('inf')):
                    g_score[nxt] = tentative
                    came_from[nxt] = current
                    heapq.heappush(open_set, (tentative + heuristic(nxt), nxt))
        return None

    def replan(self):
        if not (self.have_drone and self.have_truck):
            return
        if not self.obstacles:
            self.path = [(self.truck_x, self.truck_y)]
            self.path_index = 0
            return
        path = self.astar((self.drone_x, self.drone_y), (self.truck_x, self.truck_y))
        if path:
            self.path = path
            self.path_index = 0
        else:
            self.get_logger().warn('A* found no path this cycle -- holding, will retry.')

    def tracking_loop(self):
        if not self.is_tracking or not self.path:
            return

        while self.path_index < len(self.path) - 1:
            wx, wy = self.path[self.path_index]
            if math.hypot(wx - self.drone_x, wy - self.drone_y) < self.grid_res:
                self.path_index += 1
            else:
                break

        goal_x, goal_y = self.path[self.path_index]
        error_x = goal_x - self.drone_x
        error_y = goal_y - self.drone_y
        error_mag = math.hypot(error_x, error_y)

        if error_mag > self.max_step:
            scale = self.max_step / error_mag
            target_x = self.drone_x + error_x * scale
            target_y = self.drone_y + error_y * scale
        else:
            target_x, target_y = goal_x, goal_y

        cmd = Position()
        cmd.x = target_x
        cmd.y = target_y
        cmd.z = self.cruise_height
        cmd.yaw = 0.0
        self.publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    tracker = DroneTracker()
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()