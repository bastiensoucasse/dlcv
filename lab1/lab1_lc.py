from __future__ import print_function

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


def computeMSE(prev, curr):
    assert curr.shape == prev.shape
    return np.sum(np.square(curr.astype(np.int32) - prev.astype(np.int32))) / curr.size


def testComputeMSE():
    HEIGHT, WIDTH = 1280, 720
    BLACK = np.zeros((HEIGHT, WIDTH))
    WHITE = np.ones((HEIGHT, WIDTH)) * 255
    assert computeMSE(BLACK, BLACK) == 0
    assert computeMSE(WHITE, WHITE) == 0
    assert computeMSE(BLACK, WHITE) == np.square(255)


def computePSNR(mse):
    if mse <= 0:
        return 0
    return 10 * np.log10(np.square(255) / mse)


def testComputePSNR():
    assert computePSNR(np.square(255)) == 0


def computeEntropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    assert hist.shape == (256,)
    prob = hist[hist > 0] / img.size
    return -np.sum(prob * np.log2(prob))


def testComputeEntropy():
    HEIGHT, WIDTH = 1280, 720
    BLACK = np.zeros((HEIGHT, WIDTH))
    TEST = np.concatenate((np.zeros((int(HEIGHT / 2), WIDTH)),
                          np.ones((int(HEIGHT / 2), WIDTH)) * 255), axis=0)
    assert computeEntropy(BLACK) == 0
    assert computeEntropy(TEST) == 1


def computeOpticalFlow1(prev, curr):
    assert curr.shape == prev.shape
    return cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)


def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:, :, 0] += np.arange(w)
    map[:, :, 1] += np.arange(h)[:, np.newaxis]
    return cv2.remap(prev, map, None, cv2.INTER_LINEAR)


def computeErrorImage(im1, im2):
    assert im2.shape == im1.shape
    err = np.minimum(255, np.maximum(0, im2 - im1 + 128))
    assert err.shape == im2.shape
    return err


def computeGME(flow):
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w / 2, h / 2])
    src[:, :, 0] += np.arange(w)
    src[:, :, 1] += np.arange(h)[:, np.newaxis]
    src -= c
    srcPts = src.reshape((w * h, 2))
    dst = src + flow
    dstPts = dst.reshape((w * h, 2))
    h, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
    return cv2.perspectiveTransform(src, h) - src


def computeGMEError(flow, gme):
    flow_split, gme_split = np.dsplit(flow, 2), np.dsplit(gme, 2)
    error = np.sqrt(np.square(flow_split[0] - gme_split[0]) + np.square(flow_split[1] - gme_split[1]))
    return error / np.max(error)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read video file')
    parser.add_argument('video', help='input video filename')
    parser.add_argument('deltaT', help='input deltaT between frames', type=int)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened() == False):
        print("ERROR: unable to open video: "+args.video)
        quit()

    deltaT = args.deltaT

    previousFrames = []
    frameNumbers = []
    mses = []
    psnrs = []
    mse0s = []
    psnr0s = []
    ents = []
    entEs = []

    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if (ret == False):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (len(previousFrames) >= deltaT):
            prev = previousFrames.pop(0)

            flow = computeOpticalFlow1(prev, gray)

            compensatedFrame = computeCompensatedFrame(prev, flow)

            # cv2.imshow('compensated', compensatedFrame)

            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)

            # cv2.imshow('imErr0', imErr0)
            cv2.imshow('imErr', imErr)

            mse0 = computeMSE(prev, gray)
            psnr0 = computePSNR(mse0)
            mse = computeMSE(compensatedFrame, gray)
            psnr = computePSNR(mse)
            ent = computeEntropy(gray)
            entE = computeEntropy(imErr)

            frameNumbers.append(i)
            mses.append(mse)
            psnrs.append(psnr)
            mse0s.append(mse0)
            psnr0s.append(psnr0)
            ents.append(ent)
            entEs.append(entE)

            gme = computeGME(flow)

            gmeError = computeGMEError(flow, gme)

            # cv2.imshow('flow', draw_flow(gray, flow))
            # cv2.imshow('gme', draw_flow(gray, gme))
            cv2.imshow('gmeError', gmeError)

        previousFrames.append(gray.copy())
        i += 1

        cv2.imshow('frame', gray)

        cv2.waitKey(1)

    plt.plot(frameNumbers, mse0s, label='MSE0')
    plt.plot(frameNumbers, mses, label='MSE')
    plt.xlabel('frames')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE0 vs MSE')
    plt.savefig("mse.png")
    # plt.show()
    plt.cla()

    plt.plot(frameNumbers, ents, label='Entropy)')
    plt.plot(frameNumbers, entEs, label='EntropyE')
    plt.xlabel('frames')
    plt.ylabel('Entropy')
    plt.legend()
    plt.title('Entropy vs EntropyE')
    plt.savefig("entropy.png")
    # plt.show()
    plt.cla()

    plt.plot(frameNumbers, psnr0s, label='PSNR0')
    plt.plot(frameNumbers, psnrs, label='PSNR')
    plt.xlabel('frames')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR0 vs PSNR')
    plt.savefig("psnr.png")
    # plt.show()
    plt.cla()

    cap.release()
    cv2.destroyAllWindows()
