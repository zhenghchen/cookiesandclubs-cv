import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np
import random
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class Shape:
    CIRCLE = 'circle'
    TRIANGLE = 'triangle'
    SQUARE = 'square'
    PENTAGON = 'pentagon'
    HEXAGON = 'hexagon'
    
    @staticmethod
    def get_vertices(shape_type, center, size):
        x, y = center
        if shape_type == Shape.CIRCLE:
            return []  # Circle handled separately in drawing
        elif shape_type == Shape.TRIANGLE:
            r = size // 2
            return np.array([
                [x, y - r],  # top
                [x - r * 0.866, y + r * 0.5],  # bottom left
                [x + r * 0.866, y + r * 0.5]   # bottom right
            ], np.int32)
        elif shape_type == Shape.SQUARE:
            r = size // 2
            return np.array([
                [x - r, y - r], [x + r, y - r],
                [x + r, y + r], [x - r, y + r]
            ], np.int32)
        elif shape_type == Shape.PENTAGON:
            r = size // 2
            vertices = []
            for i in range(5):
                angle = i * 2 * math.pi / 5 - math.pi / 2
                vx = x + r * math.cos(angle)
                vy = y + r * math.sin(angle)
                vertices.append([int(vx), int(vy)])
            return np.array(vertices, np.int32)
        elif shape_type == Shape.HEXAGON:
            r = size // 2
            vertices = []
            for i in range(6):
                angle = i * 2 * math.pi / 6
                vx = x + r * math.cos(angle)
                vy = y + r * math.sin(angle)
                vertices.append([int(vx), int(vy)])
            return np.array(vertices, np.int32)
        return np.array([], np.int32)

class DragShape:
    def __init__(self, posCenter, shape_type, size=100):
        if isinstance(posCenter, list):
            self.posCenter = posCenter
        else:
            self.posCenter = [posCenter[0], posCenter[1]]
        self.targetPos = self.posCenter.copy()
        self.size = size
        self.smoothing = 0.5
        self.shape_type = shape_type
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.matched = False

    def isInBounds(self, cursor):
        cx, cy = self.posCenter
        r = self.size // 2
        # Use a circular hit box for all shapes for simplicity
        dx = cursor[0] - cx
        dy = cursor[1] - cy
        return (dx * dx + dy * dy) <= r * r

    def update(self, cursor, isPinching, activeRect):
        if activeRect == self and isPinching:
            self.targetPos = [cursor[0], cursor[1]]
            self.posCenter[0] += (self.targetPos[0] - self.posCenter[0]) * (1 - self.smoothing)
            self.posCenter[1] += (self.targetPos[1] - self.posCenter[1]) * (1 - self.smoothing)
            return self
        
        if activeRect is None:
            if isPinching and self.isInBounds(cursor):
                self.targetPos = [cursor[0], cursor[1]]
                self.posCenter[0] += (self.targetPos[0] - self.posCenter[0]) * 0.5
                self.posCenter[1] += (self.targetPos[1] - self.posCenter[1]) * 0.5
                return self
        elif activeRect != self:
            return activeRect
        
        if not isPinching and activeRect == self:
            self.targetPos = self.posCenter.copy()
            return None
            
        return activeRect

    def draw(self, img, is_outline=False):
        cx, cy = map(int, self.posCenter)
        vertices = Shape.get_vertices(self.shape_type, (cx, cy), self.size)
        
        thickness = 2 if is_outline else cv2.FILLED
        
        if self.shape_type == Shape.CIRCLE:
            cv2.circle(img, (cx, cy), self.size // 2, self.color, thickness)
        else:
            if is_outline:
                # For outlined shapes, draw a single time with consistent thickness
                cv2.polylines(img, [vertices], True, self.color, thickness)
            else:
                cv2.fillPoly(img, [vertices], self.color)
        
        if not is_outline:
            # Add a small dot in the center for better dragging feedback
            cv2.circle(img, (cx, cy), 3, (0, 0, 0), cv2.FILLED)


# Create shapes list with one of each type
shape_types = [Shape.CIRCLE, Shape.TRIANGLE, Shape.SQUARE, Shape.PENTAGON, Shape.HEXAGON]
shapes = []
target_positions = []
score = 0
total_targets = len(shape_types)

# Create shapes at the top of the screen
for i, shape_type in enumerate(shape_types):
    shapes.append(DragShape([i * 250 + 200, 120], shape_type, size=120))  # Increased size and spacing

# Create target positions randomly on the screen
for shape_type in shape_types:
    while True:
        x = random.randint(100, 1180)  # Keep away from edges
        y = random.randint(300, 620)   # Keep in lower half of screen
        
        # Check if position is far enough from other targets
        valid_pos = True
        for pos in target_positions:
            dx = x - pos[0]
            dy = y - pos[1]
            if (dx * dx + dy * dy) < 150 * 150:  # Minimum distance of 150 pixels
                valid_pos = False
                break
        
        if valid_pos:
            target_positions.append([x, y])
            break

activeRect = None
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    cursor = None
    isPinching = False

    if hands:
        hand1 = hands[0]
        
        point1 = hand1["lmList"][4][:2]   # Thumb tip (x,y)
        point2 = hand1["lmList"][8][:2]   # Index finger tip (x,y)
        distance = detector.findDistance(point1, point2, img)[0] 
        
        cursor = point2 
        isPinching = distance < 40

    if not hands or not isPinching:
        activeRect = None
    
    if cursor is not None:
        for shape in shapes:
            activeRect = shape.update(cursor, isPinching, activeRect)
    
    # Check for matches when a shape is released
    if activeRect is None:
        for i, (shape, target_pos) in enumerate(zip(shapes, target_positions)):
            if not shape.matched:  # Only check unmatched shapes
                dx = shape.posCenter[0] - target_pos[0]
                dy = shape.posCenter[1] - target_pos[1]
                distance = math.sqrt(dx * dx + dy * dy)
                
                # If shape is close to its corresponding target and they're the same type
                if distance < 40:
                    shape.matched = True
                    score += 1

    # Create a clean image for drawing
    imgNew = np.zeros_like(img, np.uint8)
    
    # Draw target positions (outlines) with consistent thickness
    for i, target_pos in enumerate(target_positions):
        temp_shape = DragShape(target_pos, shape_types[i], size=120)
        temp_shape.color = (200, 200, 200)  # Bright gray for visibility
        # Draw outline once with consistent thickness
        temp_shape.draw(imgNew, is_outline=True)
        
        # Add subtle glow effect without affecting line thickness
        kernel = np.ones((3,3), np.uint8)
        mask = imgNew.copy()
        mask = cv2.dilate(mask, kernel, iterations=1)
        # Reduce the glow intensity to maintain consistent appearance
        imgNew = cv2.addWeighted(imgNew, 1, mask, 0.2, 0)

    # Draw shapes with more vibrant colors
    for shape in shapes:
        # Make colors more vibrant
        r, g, b = shape.color
        shape.color = (min(r + 50, 255), min(g + 50, 255), min(b + 50, 255))
        shape.draw(imgNew)

    # Blend the shapes with the camera image with less transparency
    out = img.copy()
    alpha = 0.3  # Reduced alpha makes shapes more visible
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # Draw score with a more visible style
    # Draw shadow first
    cv2.putText(out, f'Score: {score}/{total_targets}', (52, 52), 
                font, 1.5, (0, 0, 0), 4)  # Larger, with shadow
    # Draw text over shadow
    cv2.putText(out, f'Score: {score}/{total_targets}', (50, 50), 
                font, 1.5, (0, 255, 100), 3)  # Brighter green

    cv2.imshow("Image", out)
    cv2.waitKey(1)