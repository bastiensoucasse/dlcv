#https://docs.opencv.org/4.5.3/d6/d00/tutorial_py_root.html
#https://docs.opencv.org/4.5.3/dd/d43/tutorial_py_video_display.html "Playing Video from file "

from __future__ import print_function

import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt


def computeMSE(prev, curr):
    assert(curr.shape == prev.shape)
    h, w = curr.shape[:2]
    return np.sum(np.square(curr.astype(np.int32) - prev.astype(np.int32))) / (h * w)

def computeMSE_test():
    black = np.zeros([10, 10])
    white = np.ones([10, 10]) * 255

    assert(computeMSE(black, black) == 0.0)
    assert(computeMSE(white, white) == 0.0)
    assert(computeMSE(white, black) == np.square(255))
    print("MSE test passed!")

    
def computePSNR(mse):
    if mse <= 0 :
        return 0
    return 10 * np.log10(np.square(255) / mse)

def computePSNR_test(): # assumes that computeMSE works correctly
    black = np.zeros([10, 10])
    white = np.ones([10, 10]) * 255
    assert(computePSNR(computeMSE(white, black)) == 0.0)
    print("PSNR test passed!")


def computeEntropy(img):
    hist, _ = np.histogram(img, 256, [0, 255])
    p = hist / img.size # probability histogram
    return -np.sum(p * np.log2(p, where=p!= 0))

def computeEntropy_test():
    assert(computeEntropy(np.zeros([10, 10])) == 0.0)
    img = np.concatenate((np.zeros([5, 5]), np.ones([5, 5])))
    assert(computeEntropy(img) == 1.0)
    print("Entropy test passed!")


def computeErrorImage(im1, im2):
    return np.minimum(255, np.maximum(0, im1 - im2 + 128))
    # return im1

def computeOpticalFlow1(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res

    
def computeGME(flow): #flow = vecteurs de déplacement
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w/2, h/2])
    src[:,:,0] += np.arange(w)
    src[:,:,1] += np.arange(h)[:,np.newaxis]
    src -= c;
    srcPts = src.reshape((h*w, 2))

    dst = src + flow
    dstPts = dst.reshape((h*w, 2))
    H, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
    gme = cv2.perspectiveTransform(src, H) - src
    return gme

def computeGMEError(flow, gme):
    h, w = flow.shape[:2]
    flowX, flowY = np.dsplit(flow, 2)
    gmeX, gmeY = np.dsplit(gme, 2)

    err = np.sqrt(np.square(flowX - gmeX) + 
                  np.square(flowY - gmeY))

    return err / np.max(err)


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


if __name__=='__main__':

    # computeMSE_test()
    # computePSNR_test()
    # computeEntropy_test()
    # assert False

    parser = argparse.ArgumentParser(description='Read video file')
    parser.add_argument('video', help='input video filename')
    parser.add_argument('deltaT', help='input deltaT between frames', type=int)

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    if (cap.isOpened() == False):
        print("ERROR: unable to open video: "+args.video)
        quit()

    deltaT=args.deltaT

    previousFrames=[]
    frameNumbers = []
    mses = []
    psnrs = []
    mse0s = []
    psnr0s = []
    ents = []
    entEs = []

    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if (ret==False):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if (len(previousFrames) >= deltaT):
            prev = previousFrames.pop(0)

            flow = computeOpticalFlow1(prev, gray)
        
        
            compensatedFrame = computeCompensatedFrame(prev, flow)

            #cv2.imshow('compensated', compensatedFrame)
        
            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)
            
            # cv2.imshow('imErr0', imErr0)
            # cv2.imshow('imErr', imErr)
            
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
            # cv2.imshow('gmeError', gmeError)

        
        previousFrames.append(gray.copy())
        i+=1

        # cv2.imshow('frame', gray)
        
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
