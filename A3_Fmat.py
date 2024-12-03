import numpy as np
import cv2
import time

from numpy.ma.core import shape

from CV_A3_P1_Data.compute_avg_reproj_error import compute_avg_reproj_error


def compute_F_raw(M):
    A = []
    for i in range(len(M)):
        x, y, x_p, y_p = M[i]
        A.append([x * x_p, x_p * y, x_p, x * y_p, y * y_p, y_p, x, y, 1])
    A = np.array(A)
    U, s, vh = np.linalg.svd(A)
    F = vh[-1].reshape([3, 3])
    return F

def get_normalize_matrix(h,w):
    h_half = h/2
    w_half = w/2
    matrixT = np.array([
        [1,0,-w_half],
        [0,1,-h_half],
        [0,0,1]
    ])
    matrixS = np.array([
        [1/w_half, 0, 0],
        [0, 1/h_half, 0],
        [0, 0, 1]
    ])
    return np.dot(matrixS, matrixT)

def compute_F_norm(M):
    global h, w
    normalize_matrix = get_normalize_matrix(h,w)
    left_norm = [np.dot(normalize_matrix, vec)[:2] / np.dot(normalize_matrix, vec)[2] for vec in np.c_[M[:,:2], np.ones(M.shape[0])]]
    right_norm = [np.dot(normalize_matrix, vec)[:2] / np.dot(normalize_matrix, vec)[2] for vec in np.c_[M[:,2:], np.ones(M.shape[0])]]
    M_norm = np.c_[left_norm, right_norm]
    # print(M_norm)
    F = compute_F_raw(M_norm)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F_p = np.dot(U, np.dot(np.diag(S), Vt))
    return np.dot(normalize_matrix.T, np.dot(F_p, normalize_matrix))


def compute_F_mine(M, num_points = 8):
    start = time.time()
    normalize_matrix = get_normalize_matrix(h,w)
    left_norm = [np.dot(normalize_matrix, vec)[:2] / np.dot(normalize_matrix, vec)[2] for vec in np.c_[M[:, :2], np.ones(M.shape[0])]]
    right_norm = [np.dot(normalize_matrix, vec)[:2] / np.dot(normalize_matrix, vec)[2] for vec in np.c_[M[:, 2:], np.ones(M.shape[0])]]
    M_norm = np.c_[left_norm, right_norm]

    min_error = 1
    min_F = None
    while time.time() - start < 4.5:
        choice = np.random.choice(np.arange(0, len(M_norm)), num_points, replace=False)
        picked = M_norm[choice]
        A = []
        for idx, (x, y, x_p, y_p) in enumerate(picked):
            A.append([x * x_p, x_p * y, x_p, x * y_p, y * y_p, y_p, x, y, 1])

        A = np.array(A)
        if np.linalg.matrix_rank(A) <8:
            continue

        _, _, vh = np.linalg.svd(A)
        F = vh[-1].reshape([3, 3])

        U, S, Vt = np.linalg.svd(F)
        S[-1] = 0
        F_p = np.dot(U, np.dot(np.diag(S), Vt))

        care = compute_avg_reproj_error(M_norm, F_p)
        if min_error > care:
            min_error = care
            min_F = F_p

    F_p = min_F
    return np.dot(np.dot(normalize_matrix.T, F_p), normalize_matrix)

def point_to_epiline(F, M):
    result = []
    M = np.c_[M, np.ones(len(M))]
    for p in M:
        result.append(np.dot(F, p))
    return result

def draw_epiline(image, l, color):
    a,b,c = l.tolist()
    height, width, _ = image.shape
    p1 = (0, int(-(c/b)))
    p2 = (width, int(-(c + (a*width))/b))
    cv2.line(image, p1, p2, color,2)

def visualize(image1, image2, F, M):
    while True:
        l_image, r_image = image1.copy(), image2.copy()
        picked = np.random.choice(np.arange(0,len(M)), size=3, replace=False)
        picked = M[picked]
        left_points = picked[:, :2]
        right_points = picked[:, 2:]

        right_lines = point_to_epiline(F, left_points)
        left_lines = point_to_epiline(F.T, right_points)

        colors = [(255,0,0),(0,255,0),(0,0,255)]
        for i in range(3):
            cv2.circle(l_image, (int(left_points[i][0]),int(left_points[i][1])), 3, colors[i], 2)
            draw_epiline(l_image, left_lines[i], colors[i])
            cv2.circle(r_image, (int(right_points[i][0]),int(right_points[i][1])), 3, colors[i],2)
            draw_epiline(r_image, right_lines[i], colors[i])

        cv2.imshow('temple',cv2.hconcat([l_image, r_image]))
        k = cv2.waitKey()
        if k == ord('q'):
            cv2.destroyAllWindows()
            return


if __name__ == '__main__':
    # ========================== Temple
    temple1 = cv2.imread('./CV_A3_P1_Data/temple1.png')
    temple2 = cv2.imread('./CV_A3_P1_Data/temple2.png')
    M_temple = np.loadtxt('./CV_A3_P1_Data/temple_matches.txt')
    h,w,_ = temple1.shape
    F_raw_temple = compute_F_raw(M_temple)
    F_norm_temple = compute_F_norm(M_temple)
    F_mine_temple = compute_F_mine(M_temple, int(len(M_temple/2)))

    print('Average Reprojection Errors (temple1.png and temple2.png)')
    print('   Raw  =',compute_avg_reproj_error(M_temple,F_raw_temple))
    print('   Norm =',compute_avg_reproj_error(M_temple, F_norm_temple))
    print('   Mine =',compute_avg_reproj_error(M_temple, F_mine_temple), end='\n\n')

    # =========================== House
    house1 = cv2.imread('./CV_A3_P1_Data/house1.jpg')
    house2 = cv2.imread('./CV_A3_P1_Data/house2.jpg')
    M_house = np.loadtxt('./CV_A3_P1_Data/house_matches.txt')
    h, w, _ = house1.shape
    F_raw_house = compute_F_raw(M_house)
    F_norm_house = compute_F_norm(M_house)
    F_mine_house = compute_F_mine(M_house, int(len(M_house)/2))

    print('Average Reprojection Errors (house1.png and house2.png)')
    print('   Raw  =', compute_avg_reproj_error(M_house, F_raw_house))
    print('   Norm =', compute_avg_reproj_error(M_house, F_norm_house))
    print('   Mine =', compute_avg_reproj_error(M_house, F_mine_house), end='\n\n')

    # =========================== Library
    library1 = cv2.imread('./CV_A3_P1_Data/library1.jpg')
    library2 = cv2.imread('./CV_A3_P1_Data/library2.jpg')
    M_library = np.loadtxt('./CV_A3_P1_Data/library_matches.txt')
    h, w, _ = library1.shape
    F_raw_library = compute_F_raw(M_library)
    F_norm_library = compute_F_norm(M_library)
    F_mine_library = compute_F_mine(M_library, int(len(M_library)/2))

    print('Average Reprojection Errors (library1.png and library2.png)')
    print('   Raw  =', compute_avg_reproj_error(M_library, F_raw_library))
    print('   Norm =', compute_avg_reproj_error(M_library, F_norm_library))
    print('   Mine =', compute_avg_reproj_error(M_library, F_mine_library), end='\n\n')

    visualize(temple1, temple2, F_mine_temple, M_temple)
    visualize(house1, house2, F_mine_house, M_house)
    visualize(library1, library2, F_mine_library, M_library)

