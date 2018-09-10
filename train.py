import os
import sys
import textwrap

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from slugify import slugify
from configuration import config
from utils import data_utils
from ved_var_attn import VarSeq2SeqVarAttnModel
from ved_det_attn import VarSeq2SeqDetAttnModel
from ded_det_attn import DetSeq2SeqDetAttnModel


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(config):
    ############### SHOW INFORMATION ###############
    print(device_lib.list_local_devices())
    wrapper = textwrap.TextWrapper(width=100)
    ############### END ###############

    ############### SETUP DIRECTORY AND RE-CONFIGURATION ###############
    current_dir = os.path.abspath('./')
    data_dir = current_dir + '/data/'
    outputs_dir = current_dir + '/outputs/'
    arch_dir = outputs_dir + slugify(config['model']) + '/'
    outputs_data_dir = outputs_dir + 'data/'

    logs_dir = arch_dir + 'summary/'
    log_str_dir = arch_dir + 'outcome/'
    model_checkpoint_dir = arch_dir + 'checkpoints/var-seq2seq-with-atten-'
    bleu_path = arch_dir + 'bleu/det-seq2seq-var-attn'
    w2v_dir = outputs_data_dir
    w2v_path = w2v_dir + 'w2v_model_news.pkl'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    if not os.path.exists(arch_dir):
        os.makedirs(arch_dir)

    print('Data dir:', data_dir)
    print('Outputs dir:', outputs_dir)
    print('Arch dir:', arch_dir)

    config['data_dir'] = data_dir
    config['logs_dir'] = logs_dir
    config['log_str_dir'] = log_str_dir
    config['model_checkpoint_dir'] = model_checkpoint_dir
    config['bleu_path'] = bleu_path
    config['wrapper'] = wrapper
    config['w2v_dir'] = w2v_dir

    input_type = 'content'
    output_type = 'title'
    decoder_filters = encoder_filters = '!"#$%&()*+/:;<=>@[\\]^`{|}~\t\n'
    ############### END ###############

    ############### DATE PREPARATION ###############
    print('[INFO] Importing the data')
    data_sources = [
        os.path.join(config['data_dir'], 'articles1.csv'),
        os.path.join(config['data_dir'], 'articles2.csv'),
        os.path.join(config['data_dir'], 'articles3.csv'),
    ]
    data = data_utils.create_news_data(
        data_sources,
        num_samples=config['num_samples'],
        preprocessing=config['preprocessing'])
    ############### END ###############

    ############### TOKENIZATION ###############
    print('[INFO] Tokenizing input and output sequences')
    input_sentences = data[input_type].values
    output_sentences = data[output_type].values

    print('Inputs:', len(input_sentences))
    print('Outputs:', len(output_sentences))

    x, word2idx_inputs, x_sen = data_utils.tokenize_sequence(
        sentences=input_sentences,
        max_num_words=config['encoder_num_tokens'],
        max_vocab_size=config['encoder_vocab'],
        filters=encoder_filters)

    y, word2idx_outputs, y_sen = data_utils.tokenize_sequence(
        sentences=output_sentences,
        max_num_words=config['decoder_num_tokens'],
        max_vocab_size=config['decoder_vocab'],
        filters=decoder_filters)
    ############### END ###############

    ############### DATA SPLITTING ###############
    print('[INFO] Split data into train-valid-test sets')
    train_data, valid_data, test_data = data_utils.create_data_split(
        x=[x, x_sen],
        y=[y, y_sen],
        valid_size=.3,
        test_size=.5,
        verbose=True)
    (x_train, y_train, x_sen_train, y_sen_train) = train_data
    (x_valid, y_valid, x_sen_valid, y_sen_valid) = valid_data
    (x_test, y_test, x_sen_test, y_sen_test) = test_data
    ############### END ###############

    ############### EMBEDDINGS AND W2V ###############
    print('[INFO] Embeddings vector and matrix')

    encoder_embeddings_matrix = data_utils.create_embedding_matrix(
        word_index=word2idx_inputs,
        embedding_dim=config['embedding_size'],
        w2v_path=w2v_path)

    decoder_embeddings_matrix = data_utils.create_embedding_matrix(
        word_index=word2idx_outputs,
        embedding_dim=config['embedding_size'],
        w2v_path=w2v_path)

    # Re-calculate the vocab size based on the word_idx dictionary
    config['encoder_vocab'] = len(word2idx_inputs)
    config['decoder_vocab'] = len(word2idx_outputs)
    ############### END ###############

    ############### TRAINING AND PREDICTION ###############
    print('############### TRAINING ###############')
    models = {
        'VarSeq2SeqDetAttnModel': VarSeq2SeqDetAttnModel,
        'VarSeq2SeqVarAttnModel': VarSeq2SeqVarAttnModel,
        'DetSeq2SeqDetAttnModel': DetSeq2SeqDetAttnModel
    }
    if not config['model'] in models:
        raise ValueError('Your model does not exist.')

    model = models[config['model']]
    model = model(
        config=config,
        encoder_embeddings_matrix=encoder_embeddings_matrix,
        decoder_embeddings_matrix=decoder_embeddings_matrix,
        encoder_word_index=word2idx_inputs,
        decoder_word_index=word2idx_outputs)
    model.train(x_train, y_train, x_valid, y_valid, y_sen_valid)
    print('############### END ###############')

    # print('############### PREDICTING AND VALIDATING ###############')
    if config['load_checkpoint'] != 0:
        checkpoint = config['model_checkpoint_dir'] + str(config['load_checkpoint']) + '.ckpt'
    else:
        checkpoint = tf.train.get_checkpoint_state(os.path.dirname('models/checkpoint')).model_checkpoint_path

    print('checkpoint:', checkpoint)

    preds = model.predict(checkpoint, x_test, y_test, y_sen_test)

    print('###### GENERATION SAMPLES ######')
    count = 5
    model.show_output_sentences(
        preds[:count],
        y_test[:count],
        x_sen_test[:count],
        y_sen_test[:count],
        '%s/%s' % (config['log_str_dir'], 'output_sentences.csv'))
    print()
    model.get_diversity_metrics(checkpoint, x_test, y_test)
    print('############### END ###############')
    ############### END ###############


if __name__ == '__main__':
    main(config)
