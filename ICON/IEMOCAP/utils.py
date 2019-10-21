import numpy as np
import pandas as pd
import pickle
from sklearn import model_selection, metrics

TEXT_EMBEDDINGS = "./IEMOCAP/data/text/IEMOCAP_text_embeddings.pickle"
VIDEO_EMBEDDINGS = "./IEMOCAP/data/video/IEMOCAP_video_features.pickle"
AUDIO_EMBEDDINGS = "./IEMOCAP/data/audio/IEMOCAP_audio_features.pickle"

trainID = pickle.load(open("./IEMOCAP/data/trainID.pkl",'rb'), encoding="latin1")
testID = pickle.load(open("./IEMOCAP/data/testID.pkl",'rb'), encoding="latin1")
valID,_ = model_selection.train_test_split(testID, test_size=.4, random_state=1227)
# valID = testID

transcripts, labels, own_historyID, other_historyID, own_historyID_rank, other_historyID_rank = pickle.load(open("./IEMOCAP/data/dataset.pkl",'rb'), encoding="latin1")

label_idx = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}


def oneHot(trainLabels, valLabels, testLabels):
	
	# Calculate the total number of classes
	numOfClasses = np.max(trainLabels)+1
	
	trainLabelOneHot = np.zeros((len(trainLabels),numOfClasses), dtype=np.float32)
	valLabelOneHot = np.zeros((len(valLabels),numOfClasses), dtype=np.float32)
	testLabelOneHot = np.zeros((len(testLabels),numOfClasses), dtype=np.float32)

	for idx, label in enumerate(trainLabels):
		trainLabelOneHot[idx, int(label)]=1.0
	for idx, label in enumerate(valLabels):
		valLabelOneHot[idx, int(label)]=1.0
	for idx, label in enumerate(testLabels):
		testLabelOneHot[idx, int(label)]=1.0

	return trainLabelOneHot, valLabelOneHot, testLabelOneHot

def updateDictText(text_transcripts_emb, text_own_history_emb, text_other_history_emb, text_emb):

	for ID, value in text_transcripts_emb.items():
		if ID in text_emb.keys():
			text_transcripts_emb[ID] = text_emb[ID]
	# updating the context faeturs
	for ID, value in text_own_history_emb.items():
		ids = own_historyID[ID]
		for idx, iD in enumerate(ids):
			if iD in text_emb.keys():
				text_own_history_emb[ID][idx]= text_emb[iD]

	# updating the context faeturs
	for ID, value in text_other_history_emb.items():
		ids = other_historyID[ID]
		for idx, iD in enumerate(ids):
			if iD in text_emb.keys():
				text_other_history_emb[ID][idx]= text_emb[iD]

	return text_transcripts_emb, text_own_history_emb, text_other_history_emb


def loadData(FLAGS):

	## Load Labels
	trainLabels = np.asarray([label_idx[labels[ID]] for ID in trainID])
	valLabels = np.asarray([label_idx[labels[ID]] for ID in valID])
	testLabels = np.asarray([label_idx[labels[ID]] for ID in testID])
	trainLabels, valLabels, testLabels = oneHot(trainLabels, valLabels, testLabels)

	## Loading Text features
	text_transcripts_emb, text_own_history_emb, text_other_history_emb = pickle.load( open(TEXT_EMBEDDINGS, 'rb'), encoding="latin1")
	if FLAGS.context:
		print("loading contextual features")
		text_emb = pickle.load(open("./IEMOCAP/data/text/IEMOCAP_text_context.pickle", 'rb'), encoding="latin1")
		text_transcripts_emb, text_own_history_emb, text_other_history_emb = updateDictText(text_transcripts_emb, text_own_history_emb, text_other_history_emb, text_emb)

	## Loading Audio features
	audio_emb = pickle.load(open(AUDIO_EMBEDDINGS, 'rb'), encoding="latin1")
	if FLAGS.context:
		audio_emb_context = pickle.load(open("./IEMOCAP/data/audio/IEMOCAP_audio_context.pickle", 'rb'), encoding="latin1")
		for ID in audio_emb.keys():
			if ID in audio_emb_context.keys():
				audio_emb[ID] = audio_emb_context[ID]
	
	## Loading Video features 
	video_emb = pickle.load(open(VIDEO_EMBEDDINGS, 'rb'), encoding="latin1")
	# video_emb_context = pickle.load(open("./IEMOCAP/data/video/IEMOCAP_video_context.pickle", 'rb'), encoding="latin1")
	# for ID in video_emb.keys():
	# 	if ID in video_emb_context.keys():
	# 		video_emb[ID] = video_emb_context[ID]
	
	## Text Embeddings for the queries
	text_trainQueries = np.asarray([text_transcripts_emb[ID] for ID in trainID])
	text_valQueries = np.asarray([text_transcripts_emb[ID] for ID in valID])
	text_testQueries = np.asarray([text_transcripts_emb[ID] for ID in testID])

	## Audio Embeddings for the queries
	audio_trainQueries = np.asarray([audio_emb[ID] for ID in trainID])
	audio_valQueries = np.asarray([audio_emb[ID] for ID in valID])
	audio_testQueries = np.asarray([audio_emb[ID] for ID in testID])

	## Video Embeddings for the queries
	video_trainQueries = np.asarray([video_emb[ID] for ID in trainID])
	video_valQueries = np.asarray([video_emb[ID] for ID in valID])
	video_testQueries = np.asarray([video_emb[ID] for ID in testID])



	
	if FLAGS.mode == "text":
		trainQueries = text_trainQueries
		valQueries = text_valQueries
		testQueries = text_testQueries
	if FLAGS.mode == "video":
		trainQueries = video_trainQueries 
		valQueries = video_valQueries
		testQueries = video_testQueries
	if FLAGS.mode == "audio":
		trainQueries = audio_trainQueries 
		valQueries = audio_valQueries 
		testQueries = audio_testQueries
	if FLAGS.mode == "textvideo":
		trainQueries = np.concatenate((text_trainQueries, video_trainQueries), axis=1)
		valQueries = np.concatenate((text_valQueries, video_valQueries), axis=1)
		testQueries = np.concatenate((text_testQueries, video_testQueries), axis=1)
	if FLAGS.mode == "audiovideo":
		trainQueries = np.concatenate((audio_trainQueries, video_trainQueries), axis=1)
		valQueries = np.concatenate((audio_valQueries, video_valQueries), axis=1)
		testQueries = np.concatenate((audio_testQueries, video_testQueries), axis=1)
	if FLAGS.mode == "textaudio":
		trainQueries = np.concatenate((text_trainQueries, audio_trainQueries), axis=1)
		valQueries = np.concatenate((text_valQueries, audio_valQueries), axis=1)
		testQueries = np.concatenate((text_testQueries, audio_testQueries), axis=1)
	if FLAGS.mode == "all":
		trainQueries = np.concatenate((text_trainQueries, audio_trainQueries, video_trainQueries), axis=1)
		valQueries = np.concatenate((text_valQueries, audio_valQueries, video_valQueries), axis=1)
		testQueries = np.concatenate((text_testQueries, audio_testQueries, video_testQueries), axis=1)

	## Pad the histories upto maximum length

	#Train queries' histories
	#(older to newer)

	trainOwnHistory = np.zeros((len(trainID), FLAGS.timesteps, trainQueries.shape[1]), dtype = np.float32)
	trainOtherHistory = np.zeros((len(trainID), FLAGS.timesteps, trainQueries.shape[1]), dtype = np.float32)
	trainOwnHistoryMask = np.zeros((len(trainID), FLAGS.timesteps), dtype = np.float32)
	trainOtherHistoryMask = np.zeros((len(trainID), FLAGS.timesteps), dtype = np.float32)

	for iddx, ID in enumerate(trainID):

		combined_historyID_rank = own_historyID_rank[ID][:] + other_historyID_rank[ID][:]

		if len(combined_historyID_rank) > 0:
		
			maxRank = np.max(combined_historyID_rank)
			own_history_rank = [maxRank - currRank for currRank in own_historyID_rank[ID]]
			other_history_rank = [maxRank - currRank for currRank in other_historyID_rank[ID]] 
			
			textOwnHistoryEmb = np.asarray(text_own_history_emb[ID])
			textOtherHistoryEmb = np.asarray(text_other_history_emb[ID])

			audioOwnHistoryEmb = np.asarray( [audio_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			audioOtherHistoryEmb = np.asarray( [audio_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )

			videoOwnHistoryEmb = np.asarray( [video_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			videoOtherHistoryEmb = np.asarray( [video_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )


			for idx, rank in enumerate(own_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						trainOwnHistory[iddx,rank] = textOwnHistoryEmb[idx]
					elif FLAGS.mode == "video":
						trainOwnHistory[iddx,rank] = videoOwnHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						trainOwnHistory[iddx,rank] = audioOwnHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						trainOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						trainOwnHistory[iddx,rank] = np.concatenate((audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						trainOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						trainOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))

						
					trainOwnHistoryMask[iddx,rank] = 1.0
			trainOwnHistory[iddx] = trainOwnHistory[iddx,::-1,:]
			trainOwnHistoryMask[iddx] = trainOwnHistoryMask[iddx,::-1]

			for idx, rank in enumerate(other_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						trainOtherHistory[iddx,rank] = textOtherHistoryEmb[idx]
					elif FLAGS.mode == "video":
						trainOtherHistory[iddx,rank] = videoOtherHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						trainOtherHistory[iddx,rank] = audioOtherHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						trainOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						trainOtherHistory[iddx,rank] = np.concatenate((audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						trainOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						trainOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))

					trainOtherHistoryMask[iddx,rank] = 1.0
			trainOtherHistory[iddx] = trainOtherHistory[iddx,::-1,:]
			trainOtherHistoryMask[iddx] = trainOtherHistoryMask[iddx,::-1]


	valOwnHistory = np.zeros((len(valID), FLAGS.timesteps, valQueries.shape[1]), dtype = np.float32)
	valOtherHistory = np.zeros((len(valID), FLAGS.timesteps, valQueries.shape[1]), dtype = np.float32)
	valOwnHistoryMask = np.zeros((len(valID), FLAGS.timesteps), dtype = np.float32)
	valOtherHistoryMask = np.zeros((len(valID), FLAGS.timesteps), dtype = np.float32)

	for iddx, ID in enumerate(valID):

		combined_historyID_rank = own_historyID_rank[ID][:] + other_historyID_rank[ID][:]

		if len(combined_historyID_rank) > 0:
		
			maxRank = np.max(combined_historyID_rank)
			own_history_rank = [maxRank - currRank for currRank in own_historyID_rank[ID]]
			other_history_rank = [maxRank - currRank for currRank in other_historyID_rank[ID]] 
			
			textOwnHistoryEmb = np.asarray(text_own_history_emb[ID])
			textOtherHistoryEmb = np.asarray(text_other_history_emb[ID])

			audioOwnHistoryEmb = np.asarray( [audio_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			audioOtherHistoryEmb = np.asarray( [audio_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )

			videoOwnHistoryEmb = np.asarray( [video_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			videoOtherHistoryEmb = np.asarray( [video_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )


			for idx, rank in enumerate(own_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						valOwnHistory[iddx,rank] = textOwnHistoryEmb[idx]
					elif FLAGS.mode == "video":
						valOwnHistory[iddx,rank] = videoOwnHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						valOwnHistory[iddx,rank] = audioOwnHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						valOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						valOwnHistory[iddx,rank] = np.concatenate((audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						valOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						valOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))

						
					valOwnHistoryMask[iddx,rank] = 1.0
			valOwnHistory[iddx] = valOwnHistory[iddx,::-1,:]
			valOwnHistoryMask[iddx] = valOwnHistoryMask[iddx,::-1]

			for idx, rank in enumerate(other_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						valOtherHistory[iddx,rank] = textOtherHistoryEmb[idx]
					elif FLAGS.mode == "video":
						valOtherHistory[iddx,rank] = videoOtherHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						valOtherHistory[iddx,rank] = audioOtherHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						valOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						valOtherHistory[iddx,rank] = np.concatenate((audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						valOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						valOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))

					valOtherHistoryMask[iddx,rank] = 1.0
			valOtherHistory[iddx] = valOtherHistory[iddx,::-1,:]
			valOtherHistoryMask[iddx] = valOtherHistoryMask[iddx,::-1]


	#Test queries' histories
	testOwnHistory = np.zeros((len(testID), FLAGS.timesteps, testQueries.shape[1]), dtype = np.float32)
	testOtherHistory = np.zeros((len(testID), FLAGS.timesteps, testQueries.shape[1]), dtype = np.float32)
	testOwnHistoryMask = np.zeros((len(testID), FLAGS.timesteps), dtype = np.float32)
	testOtherHistoryMask = np.zeros((len(testID), FLAGS.timesteps), dtype = np.float32)

	for iddx, ID in enumerate(testID):

		combined_historyID_rank = own_historyID_rank[ID][:] + other_historyID_rank[ID][:]

		if len(combined_historyID_rank) > 0:
		
			maxRank = np.max(combined_historyID_rank)
			own_history_rank = [maxRank - currRank for currRank in own_historyID_rank[ID]]
			other_history_rank = [maxRank - currRank for currRank in other_historyID_rank[ID]] 
			
			textOwnHistoryEmb = np.asarray(text_own_history_emb[ID])
			textOtherHistoryEmb = np.asarray(text_other_history_emb[ID])

			audioOwnHistoryEmb = np.asarray( [audio_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			audioOtherHistoryEmb = np.asarray( [audio_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )

			videoOwnHistoryEmb = np.asarray( [video_emb[own_historyID[ID][idx]] for idx in range(len(own_historyID[ID]))]  )
			videoOtherHistoryEmb = np.asarray( [video_emb[other_historyID[ID][idx]] for idx in range(len(other_historyID[ID]))]  )
			

			for idx, rank in enumerate(own_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						testOwnHistory[iddx,rank] = textOwnHistoryEmb[idx]
					elif FLAGS.mode == "video":
						testOwnHistory[iddx,rank] = videoOwnHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						testOwnHistory[iddx,rank] = audioOwnHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						testOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						testOwnHistory[iddx,rank] = np.concatenate((audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						testOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						testOwnHistory[iddx,rank] = np.concatenate((textOwnHistoryEmb[idx], audioOwnHistoryEmb[idx], videoOwnHistoryEmb[idx]))

					testOwnHistoryMask[iddx,rank] = 1.0
			testOwnHistory[iddx] = testOwnHistory[iddx,::-1,:]
			testOwnHistoryMask[iddx] = testOwnHistoryMask[iddx,::-1]

			for idx, rank in enumerate(other_history_rank):
				if rank < FLAGS.timesteps:
					if FLAGS.mode == "text":
						testOtherHistory[iddx,rank] = textOtherHistoryEmb[idx]
					elif FLAGS.mode == "video":
						testOtherHistory[iddx,rank] = videoOtherHistoryEmb[idx]
					elif FLAGS.mode == "audio":
						testOtherHistory[iddx,rank] = audioOtherHistoryEmb[idx]
					elif FLAGS.mode == "textvideo":
						testOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "audiovideo":
						testOtherHistory[iddx,rank] = np.concatenate((audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))
					elif FLAGS.mode == "textaudio":
						testOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx]))
					elif FLAGS.mode == "all":
						testOtherHistory[iddx,rank] = np.concatenate((textOtherHistoryEmb[idx], audioOtherHistoryEmb[idx], videoOtherHistoryEmb[idx]))

					testOtherHistoryMask[iddx,rank] = 1.0
			testOtherHistory[iddx] = testOtherHistory[iddx,::-1,:]
			testOtherHistoryMask[iddx] = testOtherHistoryMask[iddx,::-1]

	return trainQueries, trainOwnHistory, trainOtherHistory, trainOwnHistoryMask, trainOtherHistoryMask, trainLabels, \
			valQueries, valOwnHistory, valOtherHistory, valOwnHistoryMask, valOtherHistoryMask, valLabels, \
			testQueries, testOwnHistory, testOtherHistory, testOwnHistoryMask, testOtherHistoryMask, testLabels


if __name__ == "__main__":
	loadData()