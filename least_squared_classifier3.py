import numpy as np
import gzip
import matplotlib.pyplot as plt
from collections import Counter
import itertools

ti = gzip.open('train-images-idx3-ubyte.gz','r')
tl = gzip.open('train-labels-idx1-ubyte.gz', 'r')
si = gzip.open('t10k-images-idx3-ubyte.gz', 'r')
sl = gzip.open('t10k-labels-idx1-ubyte.gz', 'r')

image_size = 28 #size (in one dimension) of the images to be clustered
num_images = 60000 #number of images that we will cluster
test_num_images = 10000

ti.read(16)
buf = ti.read(image_size*image_size*num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size*image_size)
tl.read(8)
buf = tl.read(num_images)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
si.read(16)
buf = si.read(image_size*image_size*test_num_images)
test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_data = test_data.reshape(test_num_images, image_size*image_size)
sl.read(8)
buf = sl.read(test_num_images)
test_labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)

# image = np.asarray(data[0]).squeeze()
# plt.imshow(image)
# plt.show()
#
# # Rearrange data from 28x28 matrices to length 784 vectors.
# data_list = []
# for im in range(np.shape(data)[0]):
#     element = data[im,:,:,0].reshape((784)) #reshape each 2D image as a 1D vector with the same number of elements
#     data_list.append(element) #append this 1D image vector to our list of data vectors
#
# #Print the new representation of our images
# print("New representation of data:")
# print("Num images: %s | Length of each image: %s"%(np.shape(data_list)[0], np.shape(data_list)[1]))

data = data[:2000,:]
labels = labels[:2000]
test_data = test_data[:10000,:]
test_labels = test_labels[:10000]

def filterData(data, labels):
    # Filters the full data set all digits. Returns 10 matrices shaped similarly to the original data
    data_filt = [np.ones((0,np.shape(data)[1]))]*10
    for i in range(np.shape(data)[0]):
        data_filt[labels[i]] = np.concatenate((data_filt[labels[i]], data[i,:].reshape((1,784))))

    return data_filt

dfilt = filterData(data,labels)
test_data_filt = filterData(test_data, test_labels)

def biClassifier(data_filt, dig_to_compare):
    # Function takes as input the filtered version of all data (returned from filterData()), and a list of the digit to compare (eg. [4])
    # Reduces the sparsity of image data and computes indices of sparse elements, puts reduced data into an 'A' matrix, computes the beta matrix (least square coefficients)
    # Returns indices and betas

    # Compute some useful constants to be used throughout the classification
    feature_len = np.shape(data_filt[0])[1]
    thresh = int(np.shape(data_filt[dig_to_compare[0]])[0]*.05) # threshold of fewest non-zero elements over all vecs s.t. a given feature will appear in final representation.
    vec_sum = np.zeros(feature_len)  # counts for number of nonzero elements across all vecs
    data_filt_binary = np.copy(data_filt)

    # Calculate which elements do not meet the treshold
    for group in dig_to_compare:
        data_filt_binary[group] = (data_filt_binary[group] > .5).astype('uint8')
        for ifeature in range(feature_len):
            vec_sum[ifeature] += sum(data_filt_binary[group][:,ifeature])

    # Show the elements of the 28x28 image that we will use
    # plt.imshow((vec_sum>thresh).reshape((28,28)))
    # plt.show()

    # Indices of elements that had counts above threshold value
    indices = []
    for elem in range(len(vec_sum)):
        if vec_sum[elem] > thresh:
            indices.append(elem)

    # Allocate array of reduced size digit data
    nz_els = []
    labels_binary = []

    # For every group (digit data), use indices variable to select the elements we want from each vec
    for group in [j for j in range(10)]:
        for vec in range(np.shape(data_filt[group])[0]):
            nz_els.append([1] + [data_filt[group][vec, i] for i in indices])
            if group == dig_to_compare[0]:
                labels_binary.append(1)
            else:
                labels_binary.append(-1)

    # Convert to np.array
    nz_els = np.array(nz_els)
    labels_binary = np.array(labels_binary)

    # Compute betas
    ATA_inv_AT = np.matmul(np.linalg.inv(np.matmul(np.transpose(nz_els), nz_els)), np.transpose(nz_els))
    BETAS = np.matmul(ATA_inv_AT, labels_binary)

    return indices, BETAS

def testBiClassifer(test_data_filt, dig_to_compare, indices, BETAS,  data_len, mute='no'):
    # Inputs: filtered test data, list of digits to compare (eg. [8,1]), indices of sparse elements, betas
    # Tests the classifier coefficients computed in the classifier, and prints the accuracy
    # Returns: the predictions {-1, 1} and labels {-1, 1}

    els = []
    labels = []

    # filter out the sparse elements from the test data, and create appropriate labels
    for group in [k for k in range(10)]:
        for vec in range(np.shape(test_data_filt[group])[0]):
            temp = [1] + [test_data_filt[group][vec, i] for i in indices]
            els.append(temp)
            if group == dig_to_compare[0]:
                labels.append(1)
            else:
                labels.append(-1)

    els = np.array(els)

    # multiply betas with each xvec to obtain prediction. If prediction is > 0 --> digs_to_compare[0]. If < 0 --> digs_to_compare[1]
    predictions = []
    for xvec in els:
        pred_temp = np.matmul(np.transpose(BETAS), xvec)
        if pred_temp > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    tp = np.array([int(pred == 1 and lab == 1) for pred, lab in zip(predictions, labels)]).astype('uint8')
    tn = np.array([int(pred == -1 and lab == -1) for pred, lab in zip(predictions, labels)]).astype('uint8')
    fp = np.array([int(pred == 1 and lab == -1) for pred, lab in zip(predictions, labels)]).astype('uint8')
    fn = np.array([int(pred == -1 and lab == 1) for pred, lab in zip(predictions, labels)]).astype('uint8')

    error_rate = sum(fp+fn)/data_len
    if mute == 'no':
        print(f"Testing least squares binary for {dig_to_compare[0]}")
        print(f"Error rate: {error_rate}")
    return [tp, tn, fp, fn]

dig_to_compare = [4]
indices, Betas = biClassifier(dfilt, dig_to_compare)
tp, tn, fp, fn = testBiClassifer(test_data_filt, dig_to_compare, indices, Betas, len(test_labels))
print(f"True positive: {sum(tp)}")
print(f"True negative: {sum(tn)}")
print(f"False positive: {sum(fp)}")
print(f"False negative: {sum(fn)}")

def oneVerseAllClassifier(data_filt, test_data_filt, data, labels):

    # estimates = [[] for i in range(10)]
    IND = []
    BET = []
    for group in range(10):
        ind, bet = biClassifier(data_filt, [group])
        IND.append(ind)
        BET.append(bet)

    estimates = []
    for vec in range(np.shape(data)[0]):
        gx = []
        for test in range(10):
            temp = np.array([1] + [data[vec, i] for i in IND[test]])
            yh = np.matmul(np.transpose(BET[test]), temp)
            gx.append(yh)
        estimates.append(np.argmax(gx))
    np.array(estimates)

    # Print accuracy of predictions compared to actual labels
    est_analysis = np.zeros((10,10))
    num_correct = 0
    total = 0
    for q in range(10):
        for qq in range(10):
            est_analysis[q,qq] = sum([int(pred == q and lab == qq) for pred, lab in zip(estimates, labels)])
            total += est_analysis[q,qq]
            if q == qq:
                num_correct += est_analysis[q,qq]

    correct = [estimates[g]==labels[g] for g in range(len(estimates))]

    print(f"Error rate: {1-(num_correct/total)}")

    return est_analysis


oneVerseAllClassifier(dfilt, test_data_filt, data, labels)

def oneVerseOneClassifier(data_filt, digs_to_compare):
    # Function takes as input the filtered version of all data (returned from filterData()), and a list of the two digits to compare (eg. [8, 1])
    # Reduces the sparsity of image data and computes indices of sparse elements, puts reduced data into an 'A' matrix, computes the beta matrix (least square coefficients)
    # Returns indices and betas

    # Compute some useful constants to be used throughout the classification
    feature_len = np.shape(data_filt[0])[1]
    thresh = int((np.shape(data_filt[digs_to_compare[0]])[0]+np.shape(data_filt[digs_to_compare[1]])[0])*.4) # threshold of fewest non-zero elements over all vecs s.t. a given feature will appear in final representation.
    vec_sum = np.zeros(feature_len)  # counts for number of nonzero elements across all vecs
    data_filt_binary = np.copy(data_filt)

    # Calculate which elements do not meet the treshold
    for group in digs_to_compare:
        data_filt_binary[group] = (data_filt_binary[group] > .5).astype('uint8')
        for ifeature in range(feature_len):
            vec_sum[ifeature] += sum(data_filt_binary[group][:,ifeature])

    # Show the elements of the 28x28 image that we will use
    # plt.imshow((vec_sum>thresh).reshape((28,28)))
    # plt.show()

    # Indices of elements that had counts above threshold value
    indices = []
    for elem in range(len(vec_sum)):
        if vec_sum[elem] > thresh:
            indices.append(elem)

    # Allocate array of reduced size digit data
    nz_els = []
    labels_binary = []

    # For every group (digit data), use indices variable to select the elements we want from each vec
    for group in digs_to_compare:
        for vec in range(np.shape(data_filt[group])[0]):
            nz_els.append([1] + [data_filt[group][vec, i] for i in indices])
            if group == digs_to_compare[0]:
                labels_binary.append(1)
            elif group == digs_to_compare[1]:
                labels_binary.append(-1)

    # Convert to np.array
    nz_els = np.array(nz_els)
    labels_binary = np.array(labels_binary)

    # Compute betas
    ATA_inv_AT = np.matmul(np.linalg.inv(np.matmul(np.transpose(nz_els), nz_els)), np.transpose(nz_els))
    BETAS = np.matmul(ATA_inv_AT, labels_binary)

    return indices, BETAS

def testOneVerseOneClassifer(test_data, indices, BETAS, comparison_set, return_as_digits_voted_for=True):
    # Inputs: filtered test data, list of digits to compare (eg. [8,1]), indices of sparse elements, betas
    # Tests the classifier coefficients computed in the classifier, and prints the accuracy
    # Returns: the predictions {-1, 1} and labels {-1, 1}

    els = []
    # labels = []

    # filter out the sparse elements from the test data, and create appropriate labels
    for vec in range(np.shape(test_data)[0]):
        temp = [1] + [test_data[vec, i] for i in indices]
        els.append(temp)
            # if group == digs_to_compare[0]:
            #     labels.append(1)
            # elif group == digs_to_compare[1]:
            #     labels.append(-1)

    els = np.array(els)

    # multiply betas with each xvec to obtain prediction. If prediction is > 0 --> digs_to_compare[0]. If < 0 --> digs_to_compare[1]
    predictions = []
    for xvec in els:
        pred_temp = np.matmul(np.transpose(BETAS), xvec)
        if pred_temp > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    # Print accuracy of predictions compared to actual labels
    # yhat = np.array([pred == lab for pred, lab in zip(predictions, labels)]).astype('uint8')
    if return_as_digits_voted_for == True:
        for pred in range(len(predictions)):
            if predictions[pred] == 1:
                predictions[pred] = comparison_set[0]
            elif predictions[pred] == -1:
                predictions[pred] = comparison_set[1]
            else:
                print("Error")
                assert(False)

    return predictions

def oneVerseOneClassifierAllCombs(data_filt, test_data, test_labels):
    # Create all combinations of pairs of digits
    sets = list(itertools.combinations([i for i in range(10)], 2))

    # Allocate an array for which
    predictions_master = np.zeros((10000, 45))
    print(sets)
    for set in range(np.shape(sets)[0]):
        # For each pair of digits that we will run comparisons on, find the indices of nonzero elements, and the Betas
        ind, bet = oneVerseOneClassifier(data_filt, list(sets[set]))
        # Make the prediction using these coefficients on every vector, and store this as a column in predictions_master
        predictions_master[:,set] = np.array(testOneVerseOneClassifer(test_data, ind, bet, list(sets[set])))

    final_estimates = []
    # Loop over each vector and determine which number had the most "votes"
    for test_vec in range(np.shape(predictions_master)[0]):
        t = Counter(predictions_master[test_vec,:])
        final_estimates.append(int(t.most_common(1)[0][0]))

    correct = [int(final_estimates[i] == test_labels[i]) for i in range(len(test_labels))]
    print("Percent correct for one vs one classification:")
    print(f"{sum(correct)/len(correct)}")

    return predictions_master

predictions = oneVerseOneClassifierAllCombs(dfilt, test_data, test_labels)

confusion_matrix = np.zeros((10,10))
for i in range(len(test_data)):
    most_comm = Counter(predictions[i, :]).most_common()
    for j in range(len(most_comm)):
        confusion_matrix[test_labels[i], int(most_comm[j][0])] += Counter(predictions[i, :]).most_common()[j][1]

print(confusion_matrix)