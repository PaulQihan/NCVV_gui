import matplotlib.pyplot as plt
import numpy as np

def curve_deform():
      quality=[99,100,98,96,94,92,90]
      psnr=[45.61523364384969,45.438563950856526,  42.54448358747694, 39.628699694739446,38.19920097986857,37.36505366431342,
            36.72]
      storge=[2201,2054,1418,838,597,462,376]

      psnr=np.asarray(psnr)
      storge=np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge,psnr,"r")


      #1e-2 res
      psnr = [45.539606, 45.277340, 42.873834, 40.323965, 39.015534,
              38.155846,
              37.48]
      storge = [1117, 1060, 675, 373, 254, 193, 153]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"g")

      #1e-2 deform
      psnr = [45.43, 45.13, 42.70, 40.12, 38.78,
              37.89,
              37.20]
      storge = [998, 890 , 580, 310, 207, 154, 119]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"b")

      plt.show()
def curve1():
      quality=[99,100,98,96,94,92,90]
      psnr=[45.61523364384969,45.438563950856526,  42.54448358747694, 39.628699694739446,38.19920097986857,37.36505366431342,
            36.72]
      storge=[2201,2054,1418,838,597,462,376]

      psnr=np.asarray(psnr)
      storge=np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge,psnr,"r")

      # 1e-1
      psnr = [45.388441, 45.172868, 42.576794, 40.011031, 38.737315,
              37.848065,
              37.184370]
      storge = [709, 693, 423, 230, 156, 118, 94]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"g")

      #1e-2
      psnr = [45.539606, 45.277340, 42.873834, 40.323965, 39.015534,
              38.155846,
              37.48]
      storge = [1117, 1060, 675, 373, 254, 193, 153]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"b")

      #1e-3
      psnr = [45.67, 45.36, 43.11, 40.56, 39.20,
              38.31,
              37.69]
      storge = [1743, 1637 , 1090, 625, 435, 333, 269]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"y")

      plt.show()

def curve_gt():
      quality=[99,100,98,96,94,92,90]
      psnr=[38.49,38.45,  37.51, 35.99,35.03,34.42,
            33.97]
      storge=[2201,2054,1418,838,597,462,376]

      psnr=np.asarray(psnr)
      storge=np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge,psnr,"r")

      # 1e-1
      psnr = [36.79,36.77, 35.99, 34.69, 33.88,
              33.28,
              32.79]
      storge = [709, 693, 423, 230, 156, 118, 94]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"g")

      #1e-2
      psnr = [38.09, 38.07, 37.12, 35.58, 34.63,
              33.94,
              33.41]
      storge = [1117, 1060, 675, 373, 254, 193, 153]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("PSNR")

      plt.plot(storge, psnr,"b")

      #1e-3
      psnr = [ 38.75, 38.72, 37.74, 36.08,34.99,
              34.26,
              33.75]
      storge = [1743, 1637 , 1090, 625, 435, 333, 269]

      psnr = np.asarray(psnr)
      storge = np.asarray(storge)
      plt.xlabel("Storge(KB)")
      plt.ylabel("GT PSNR")

      plt.plot(storge, psnr,"y")

      plt.show()

def curve823():
      #统一选第10帧的
      lambda_l1 = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
      train_psnr=[27.34,31.28,34.04,34.95,35.38,35.05,35.02]
      l1_loss=[0.00157,0.000180545,0.000066635,0.000022679,0.000006766,0.000001124,0.000000120]
      storge=[464, 711,1126,1692,2087,2130,2131]
      psnr = np.asarray(train_psnr)
      storge = np.asarray(storge)
      l1_loss = np.asarray(l1_loss)
      lambda_l1= np.asarray(lambda_l1)

      # plt.xlabel("Storge(KB)")
      # plt.ylabel("Train PSNR")
      #
      # plt.plot(storge, psnr, 'o')

      plt.xlabel("lambda_la")
      plt.ylabel("train psnr")

      plt.plot(lambda_l1, psnr, 'o')
      plt.show()

curve_deform()