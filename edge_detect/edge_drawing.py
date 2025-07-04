import cv2
import numpy as np
import collections

EDGE_VER, EDGE_HOR, EDGE_IMPOSSIBLE = 0, 1, 2

def edge_detection(image_path, anchor_threshold=8, scan_interval=4, gradient_threshold=36, blur_size=5, blur_sigma=1.0, min_edge_length=10):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape

    # Blur Image
    img = cv2.GaussianBlur(img, (blur_size, blur_size), blur_sigma)

    # Gradient map and Edge direction map
    dx = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3))
    dy = np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3))

    Gradient_value = dx + dy
    Gradient_direction = np.where(dx > dy, EDGE_VER, EDGE_HOR)
    Gradient_direction[Gradient_value < gradient_threshold] = EDGE_IMPOSSIBLE

    # Add Edge Boundary
    Gradient_direction = np.pad(Gradient_direction, ((1, 1), (1, 1)), mode='constant', constant_values=EDGE_IMPOSSIBLE)
    Gradient_value = np.pad(Gradient_value, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    rows, cols = rows + 2, cols + 2

    # Extract Anchor Points
    anchors = []
    for i in range(2, rows - 2, scan_interval):
        for j in range(2, cols - 2, scan_interval):
            isanchor = False
            if Gradient_direction[i, j] == EDGE_HOR:
                mask1 = Gradient_value[i, j] - Gradient_value[i - 1, j] >= anchor_threshold
                mask2 = Gradient_value[i, j] - Gradient_value[i + 1, j] >= anchor_threshold
                isanchor = mask1 and mask2
            elif Gradient_direction[i, j] == EDGE_VER:
                mask1 = Gradient_value[i, j] - Gradient_value[i, j - 1] >= anchor_threshold
                mask2 = Gradient_value[i, j] - Gradient_value[i, j + 1] >= anchor_threshold
                isanchor = mask1 and mask2

            if isanchor:
                anchors.append((i, j))

    anchors = sorted(anchors, key=lambda x: Gradient_value[x[0], x[1]], reverse=True)

    # Smart Route DFS
    visited = np.zeros((rows, cols), dtype=bool)
    edges = np.zeros((rows, cols), dtype=np.uint8)

    def visit(x, y, visited):
        def getmaxG(p1, p2, p3):
            G1, G2, G3 = Gradient_value[p1], Gradient_value[p2], Gradient_value[p3]
            if G1 >= G2 and G1 >= G3:
                return p1
            elif G3 >= G2 and G3 >= G1:
                return p3
            else:
                return p2

        stack = collections.deque([((x, y), None)])
        current_edge = []

        while stack:
            nowp, fromdirection = stack.pop()
            nowx, nowy = nowp

            if Gradient_direction[nowx, nowy] == EDGE_IMPOSSIBLE:
                continue
            if visited[nowx, nowy]:
                continue

            visited[nowx, nowy] = True
            current_edge.append((nowx, nowy))

            if Gradient_direction[nowx, nowy] == EDGE_HOR:
                if fromdirection != 'RIGHT':
                    x1, y1 = nowx - 1, nowy - 1
                    x2, y2 = nowx, nowy - 1
                    x3, y3 = nowx + 1, nowy - 1
                    newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                    stack.append((newp, 'LEFT'))

                if fromdirection != 'LEFT':
                    x1, y1 = nowx - 1, nowy + 1
                    x2, y2 = nowx, nowy + 1
                    x3, y3 = nowx + 1, nowy + 1
                    newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                    stack.append((newp, 'RIGHT'))

            elif Gradient_direction[nowx, nowy] == EDGE_VER:
                if fromdirection != 'DOWN':
                    x1, y1 = nowx - 1, nowy - 1
                    x2, y2 = nowx - 1, nowy
                    x3, y3 = nowx - 1, nowy + 1
                    newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                    stack.append((newp, 'UP'))

                if fromdirection != 'UP':
                    x1, y1 = nowx + 1, nowy - 1
                    x2, y2 = nowx + 1, nowy
                    x3, y3 = nowx + 1, nowy + 1
                    newp = getmaxG((x1, y1), (x2, y2), (x3, y3))
                    stack.append((newp, 'DOWN'))

        return current_edge

    contours = []
    for anchorx, anchory in anchors:
        if not visited[anchorx, anchory]:
            edge = visit(anchorx, anchory, visited)
            if len(edge) >= min_edge_length:
                contours.append(edge)
                for point in edge:
                    edges[point] = 255

    #cv2.imshow('Edges', edges)

    return contours,edges