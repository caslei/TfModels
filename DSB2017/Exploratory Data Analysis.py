# From the website:
#https://www.kaggle.com/anokas/data-science-bowl-2017/exploratory-data-analysis-4

# Data Science Bowl 2017

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os
import glob



# import seaborn as sns
# p = sns.color_palette()

os.listdir('../input')

# >>> ['sample_images', 'stage1_sample_submission.csv', 'stage1_labels.csv']




for d in os.listdir('../input/sample_images'):
    print("Patient '{}' has {} scans".format(d, len(os.listdir('../input/sample_images/' + d))))
print('----')
print('Total patients {} Total DCM files {}'.format(
	  	   	     len(os.listdir('../input/sample_images')), 
	  	  	     len(glob.glob('../input/sample_images/*/*.dcm'))) )

#>>> Patient '0c9d8314f9c69840e25febabb1229fa4' has 221 scans
#>>> ...
#>>> ----
#>>> Total patients 20 Total DCM files 3604




patient_sizes = [len(os.listdir('../input/sample_images/' + d)) \
					for d in os.listdir('../input/sample_images')]
plt.figure()
plt.hist(patient_sizes)
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient')

#>>> <matplotlib.text.Text at 0x7fae5ff45198>




sizes = [os.path.getsize(dcm)/1000000 for dcm in glob.glob('../input/sample_images/*/*.dcm')]
print('DCM file sizes: min {:.3}MB max {:.3}MB avg {:.3}MB std {:.3}MB'.format(
                                np.min(sizes), np.max(sizes), np.mean(sizes), np.std(sizes)))

#>>> DCM file sizes: min 0.525MB max 0.526MB avg 0.525MB std 3.74e-05MB




df_train = pd.read_csv('../input/stage1_labels.csv')
df_train.head()

#>>> id									  cancer
#>>> 0	 0015ceb851d7251b8f399e39779d1e7d	1
#>>> 1	 0030a160d58723ff36d73f41b170ec21	0




print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))

#>>> Number of training patients: 1397
#>>> Cancer rate: 25.91%




from sklearn.metrics import log_loss
logloss = log_loss(df_train.cancer, 
				   np.zeros_like(df_train.cancer) + df_train.cancer.mean())
print('Training logloss is {}'.format(logloss))

#>>> Training logloss is 0.5721414894789518




sample = pd.read_csv('../input/stage1_sample_submission.csv')
sample['cancer'] = df_train.cancer.mean()
sample.to_csv('naive_submission.csv', index=False)




targets = df_train['cancer']
plt.figure()
plt.plot(pd.rolling_mean(targets, window=10), label='Sliding Window 10')
plt.plot(pd.rolling_mean(targets, window=50), label='Sliding Window 50')
plt.xlabel('rowID')
plt.ylabel('Mean cancer')
plt.title('Mean target over rowID - sliding mean')
plt.legend()

#>>> <matplotlib.legend.Legend at 0x7fae5ed89c18>




print('Accuracy predicting no cancer: {}%'.format((df_train['cancer'] == 0).mean()))
print('Accuracy predicting with last value: {}%'.format( \
			(df_train['cancer'] == df_train['cancer'].shift()).mean()))

#>>> Accuracy predicting no cancer: 0.740873299928418%
#>>> Accuracy predicting with last value: 0.6148890479599141%




sample = pd.read_csv('../input/stage1_sample_submission.csv')
sample.head()

#>>> id									  cancer
#>>> 0	 0015ceb851d7251b8f399e39779d1e7d  0.5
#>>> 1	 0030a160d58723ff36d73f41b170ec21  0.5




print('The test file has {} patients'.format(len(sample)))

#>>> The test file has 198 patients






import dicom


dcm = '../input/sample_images/0a38e7597ca26f9374f8ea2770ba870d/' + 
						  '4ec5ef19b52ec06a819181e404d37038.dcm'
print('Filename: {}'.format(dcm))
dcm = dicom.read_file(dcm)

#>>> Filename: ../input/sample_images/0a38e7597ca26f9374f8ea2770ba
#>>>                    870d/4ec5ef19b52ec06a819181e404d37038.dcm




# retrieve a image as a numpy array by calling 'dcm.pixel_array'
img = dcm.pixel_array	# numpy array
img[img == -2000] = 0

plt.axis('off')
plt.imshow(img)
plt.show()

plt.axis('off')
plt.imshow(-img) # Invert colors with -
plt.show()





def dicom_to_image(filename):
    dcm = dicom.read_file(filename)
    img = dcm.pixel_array # numpy array
    img[img == -2000] = 0
    return img

files = glob.glob('../input/sample_images/*/*.dcm')

f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
for i in range(20):
    plots[i // 5, i % 5].axis('off')
    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files)), cmap=plt.cm.bone)




def get_slice_location(dcm):
    return float(dcm[0x0020, 0x1041].value) # specified by dicom format

# Returns a list of images for that patient_id, in ascending order of Slice Location
def load_patient(patient_id):
    files = glob.glob('../input/sample_images/{}/*.dcm'.format(patient_id))
    imgs = {}
    for f in files:
        dcm = dicom.read_file(f)
        img = dcm.pixel_array	# numpy array
        img[img == -2000] = 0
        sl = get_slice_location(dcm)
        imgs[sl] = img
        
    # Not a very elegant way to do this
    sorted_imgs = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]
    return sorted_imgs




pat = load_patient('0a38e7597ca26f9374f8ea2770ba870d')  
f, plots = plt.subplots(11, 10, sharex='all', sharey='all', figsize=(10, 11))
for i in range(110):
    plots[i // 10, i % 10].axis('off')
    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone) 



pat = load_patient('0acbebb8d463b4b9ca88cf38431aac69')
f, plots = plt.subplots(21, 10, sharex='all', sharey='all', figsize=(10, 21))
for i in range(203):
    plots[i // 10, i % 10].axis('off')
    plots[i // 10, i % 10].imshow(pat[i], cmap=plt.cm.bone)     



# This function takes in a single frame from the DICOM and returns a single frame in RGB format.
def normalise(img):
    normed = (img / 14).astype(np.uint8) # Magic number, scaling to create int between 0 and 255
    img2 = np.zeros([*img.shape, 3], dtype=np.uint8)
    for i in range(3):
        img2[:, :, i] = normed
    return img2



npat = [normalise(p) for p in pat]
pat = load_patient('0acbebb8d463b4b9ca88cf38431aac69')

import matplotlib.animation as animation
def animate(pat, gifname):
    # Based on @Zombie's code
    fig = plt.figure()
    anim = plt.imshow(pat[0], cmap=plt.cm.bone)
    def update(i):
        anim.set_array(pat[i])
        return anim,
    
    a = animation.FuncAnimation(fig, update, frames=range(len(pat)), interval=50, blit=True)
    a.save(gifname, writer='imagemagick')
    
animate(pat, 'test.gif')        