import numpy as np, pandas as pd, tensorflow_hub as hub, tensorflow_text as text, tensorflow as tf, tensorflow.keras.backend as K
from glob import glob

max_sequence = 512
target_map = {0: 'nothing', 1: 'Lead', 2: 'Position', 3: 'Claim', 4: 'Evidence', 5: 'Counterclaim', 6: 'Rebuttal', 7: 'Concluding Statement'}
sample = pd.DataFrame(columns=['id', 'text', 'word index', 'length'])
test_txt = glob('test/*.txt')
for file in test_txt:
	with open(file, 'r') as text_file:
		text_id = file.split('/')[-1].replace('.txt', '')
		discourse_text = text_file.read().split()
		index = 0
		while discourse_text:
			size = sample.index.size
			length = max_sequence if max_sequence < len(discourse_text) else len(discourse_text)
			sample.loc[size] = [text_id, ' '.join(discourse_text[:length]), index, length]
			discourse_text = discourse_text[length:]
			index += max_sequence

def macro_f1(y_true, y_pred, beta=1, threshold=.5):
	y_true = K.cast(y_true, 'float')
	y_pred = tf.argmax(y_pred, -1)
	y_pred = K.cast(K.greater(K.cast(y_pred, 'float'), threshold), 'float')

	tp = K.sum(y_true * y_pred, axis=0)
	fp = K.sum((1 - y_true) * y_pred, axis=0)
	fn = K.sum(y_true * (1 - y_pred), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + K.epsilon())
	f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
	return K.mean(f1)

preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
tokenize = hub.KerasLayer(preprocessor.tokenize)
bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=max_sequence))
encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2', trainable=True)
def build_model(bert_layer):
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
	tokenized_input = [tokenize(text_input)]
	encoder_inputs = bert_pack_inputs(tokenized_input)
	sequence_output = bert_layer(encoder_inputs)['sequence_output']
	out = tf.keras.layers.Dense(len(target_map), activation='softmax')(sequence_output)

	model = tf.keras.Model(inputs=text_input, outputs=out)
	model.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=[macro_f1])
	return model

model = build_model(encoder)
model.load_weights('checkpoint')
prediction = model.predict(sample['text'])
result = pd.DataFrame()
for split in range(prediction.shape[0]):
	prediction_string = dict()
	first_word = sample.loc[split, 'word index']
	text_length = sample.loc[split, 'length']
	for word_index in range(prediction[split].shape[0]):
		if word_index < text_length:
			discourse_type = target_map[np.argmax(prediction[split][word_index])]
			if prediction_string.get(discourse_type):
				prediction_string[discourse_type].append(first_word + word_index)
			else:
				prediction_string[discourse_type] = [first_word + word_index]
	for key, value in prediction_string.items():
		size = result.index.size
		result.loc[size, 'id'], result.loc[size, 'class'], result.loc[size, 'predictionstring'] = sample.loc[split, 'id'], key, ' '.join([str(item) for item in value])

result = result.groupby(['id', 'class'])['predictionstring'].apply(lambda item: item.str.cat(sep=' ')).reset_index()
result.drop(result[result['class'] == 'nothing'].index, inplace=True)
result.to_csv('submission.csv', index=False)