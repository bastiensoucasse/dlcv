
#https://docs.opencv.org/4.5.3/d6/d00/tutorial_py_root.html
#https://docs.opencv.org/4.5.3/dd/d43/tutorial_py_video_display.html "Playing Video from file"

from __future__ import print_function

import argparse
import numpy as np
import cv2

import matplotlib.pyplot as plt


def computeMSE(prev, curr): #version with OpenCV absdiff
    m = cv2.absdiff(prev, curr)
    m = m.astype(np.int32)
    m = np.square(m)
    s = m.sum() / m.size
    return s

def computeMSE1(prev, curr): #No OpenCV/only numpy
    mse = np.sum((curr.astype(np.int32)-prev.astype(np.int32))**2) / (curr.shape[0]*curr.shape[1])
    return mse

def computeMSE2(prev, curr):  #No OpenCV/only numpy
    p = prev.astype(np.int32)
    c = curr.astype(np.int32)
    m = np.square(p-c)
    s = m.sum() / m.size
    return s

def computeMSE3(prev, curr):  #No OpenCV/only numpy
    mse = 0
    M, N = prev.shape
    MN = float(M*N)
    error2 = np.power(np.float32(curr) - np.float32(prev), 2)
    mse = np.sum(error2)/MN
    return mse

def computeMSEbug1(prev, curr):  #BUG: the computation overflows the element type
    mse = 0
    (M,N) = prev.shape ;
    x = (curr-prev)**2 ;
    mse = (1.0/(M*N))*x.sum() ;
    return mse


def computePSNR(mse):
    if (mse > 0):
        return 10 * np.log10((255*255) / mse)
    else:
        return 0

def computeEntropy(img): #with numpy histogram() & python "list comprehension"
    h, w = img.shape[:2]
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(np.float32) / (w*h)
    loghist = np.log2(np.array([1 if x==0 else x for x in hist]))
    m = hist*loghist
    ent = - m.sum()
    return ent

def computeEntropyA(img): #with numpy histogram() & python "list comprehension" [faster: log2 only computed on non zero]
    h, w = img.shape[:2]
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    hist = hist.astype(np.float32) / (w*h)
    loghist = np.array([0 if x==0 else np.log2(x) for x in hist])
    m = hist*loghist
    ent = - m.sum()
    return ent

def computeEntropy0(img): #with numpy histogram() & for loop
    p = np.histogram(img[0], bins=np.arange(256), density=True)[0]
    for i in range(p.shape[0]-1):
        if (p[i] != 0):
            p[i] = p[i]*np.log2(p[i])
    ent = -np.sum(p)
    return ent

def computeEntropy1(img): #with numpy unique()
    value, counts = np.unique(img, return_counts=True)
    norm_counts = counts / counts.sum()
    ent = -(norm_counts * np.log(norm_counts)/np.log(2)).sum()
    return ent

def computeEntropy2(img):#with numpy histogram() and nonzero()/take()
    hist, bin_edges = np.histogram(img, bins=np.arange(256))
    N,M = img.shape[:2]
    hist = hist/(N*M)
    hist = np.take(hist, np.nonzero(hist))[0]
    logProbability = np.multiply(hist, np.log2(hist))
    ent = -np.sum(logProbability)
    return ent

def computeEntropy3(img): #with OpenCV calcHist()
    prob = cv2.calcHist([img],[0],None,[256],[0,256]) / (img.shape[0]*img.shape[1])
    prob = list(filter(lambda p: p > 0, np.ravel(prob)))
    ent = -np.sum(np.multiply(prob,np.log2(prob)))
    return ent


    
def computeOpticalFlow1(prev, curr):
    #flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow = cv2.calcOpticalFlowFarneback(curr, prev, flow=None, pyr_scale=0.5, levels=3, winsize=20, iterations=15, poly_n=5, poly_sigma=1.2, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow

def computeCompensatedFrame(prev, flow):
    h, w = flow.shape[:2]
    map = flow.copy()
    map[:,:,0] += np.arange(w)
    map[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(prev, map, None, cv2.INTER_LINEAR)
    return res

def computeErrorImage(im1, im2):
    res = cv2.addWeighted(im1, 1, im2, -1, 128)
    return res
    
def computeGME(flow):
    src = np.zeros_like(flow)
    h, w = flow.shape[:2]
    c = np.array([w/2, h/2])
    src[:,:,0] += np.arange(w)
    src[:,:,1] += np.arange(h)[:,np.newaxis]
    src -= c;

    dst = src + flow
   
    srcPts = src.reshape((h*w, 2)) 
    dstPts = dst.reshape((h*w, 2))

    #https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html#gafd3ef89257e27d5235f4467cbb1b6a63
    hom, mask = cv2.findHomography(srcPts, dstPts, cv2.RANSAC)
    
    #https://docs.opencv.org/4.5.3/d2/de8/group__core__array.html#gad327659ac03e5fd6894b90025e6900a7
    dst2 = cv2.perspectiveTransform(src, hom)

    gme = dst2-src
    
    return gme

def computeGMEError(flow, gme):
    err = np.sqrt(np.square(flow[:,:,0]-gme[:,:,0]) + np.square(flow[:,:,1]-gme[:,:,1]))
    return err

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

            # cv2.imshow('compensated', compensatedFrame)
        
            imErr0 = computeErrorImage(prev, gray)
            imErr = computeErrorImage(compensatedFrame, gray)
            
            # cv2.imshow('imErr0', imErr0)
            # cv2.imshow('imErr', imErr)
            
            mse0 = computeMSE(prev, gray)
            mse1 = computeMSE1(prev, gray)
            mse2 = computeMSE2(prev, gray)
            mse3 = computeMSE3(prev, gray)
            mse1bug = computeMSEbug1(prev, gray)
            psnr0 = computePSNR(mse0)
            mse = computeMSE(compensatedFrame, gray)
            psnr = computePSNR(mse)
            ent = computeEntropy(gray)
            entE = computeEntropy(imErr)

            entA = computeEntropyA(imErr)
            ent1 = computeEntropy1(imErr)
            ent2 = computeEntropy2(imErr)
            ent3 = computeEntropy3(imErr)

            # print("mse0=", mse0, "=? ", mse1, "=? ", mse2, "=? ", mse3, "=? ", mse1bug)
            # print("entE=", entE, "=? ", entA, "=? ", ent1, "=? ", ent2, "=? ", ent3)
        
            frameNumbers.append(i)
            mses.append(mse)
            psnrs.append(psnr)
            mse0s.append(mse0)
            psnr0s.append(psnr0)
            ents.append(ent)
            entEs.append(entE)

            # print('{} {}'.format(ent, entE))
        
            
            gme = computeGME(flow)
            
            gmeError = computeGMEError(flow, gme)
            
            #cv2.imshow('flow', draw_flow(gray, flow))
            #cv2.imshow('gme', draw_flow(gray, gme))

            #cv2.imshow('gmeError', gmeError)

        
        previousFrames.append(gray.copy())
        i+=1

        #cv2.imshow('frame', gray)
        
        #cv2.waitKey(1)

    
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
