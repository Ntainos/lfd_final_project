import argparse, os, json, numpy as np, pandas as pd, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils_data import read_tsv, prepare_dataframe
from utils_eval import full_report

def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def build_model(vocab_size, emb_dim=100, max_len=64):
    inputs = keras.Input(shape=(max_len,), dtype='int32')
    emb = layers.Embedding(vocab_size, emb_dim)(inputs)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(emb)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--max_len', type=int, default=64)
    ap.add_argument('--emb_dim', type=int, default=100)
    ap.add_argument('--normalize_twitter', type=int, default=1)
    ap.add_argument('--mask_lexicon', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seeds(args.seed)

    train = read_tsv(args.train)
    dev = read_tsv(args.dev)
    test = read_tsv(args.test)

    train, text_col, label_col = prepare_dataframe(train, bool(args.normalize_twitter), bool(args.mask_lexicon))
    dev, _, _ = prepare_dataframe(dev, bool(args.normalize_twitter), bool(args.mask_lexicon))
    test, _, _ = prepare_dataframe(test, bool(args.normalize_twitter), bool(args.mask_lexicon))

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tok = Tokenizer(num_words=40000, oov_token="<unk>")
    tok.fit_on_texts(pd.concat([train[text_col], dev[text_col], test[text_col]]).tolist())
    X_train = pad_sequences(tok.texts_to_sequences(train[text_col].tolist()), maxlen=args.max_len, padding='post', truncating='post')
    X_dev = pad_sequences(tok.texts_to_sequences(dev[text_col].tolist()), maxlen=args.max_len, padding='post', truncating='post')
    X_test = pad_sequences(tok.texts_to_sequences(test[text_col].tolist()), maxlen=args.max_len, padding='post', truncating='post')

    y_train = pd.Series(train[label_col]).map({'NOT':0,'OFF':1}).values
    y_dev = pd.Series(dev[label_col]).map({'NOT':0,'OFF':1}).values

    model = build_model(vocab_size=len(tok.word_index)+1, emb_dim=args.emb_dim, max_len=args.max_len)
    model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=args.epochs, batch_size=64, verbose=2)

    y_dev_pred = model.predict(X_dev, batch_size=256).argmax(axis=1)
    rep = full_report(['NOT' if y==0 else 'OFF' for y in y_dev], ['NOT' if y==0 else 'OFF' for y in y_dev_pred])
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/lstm_dev_report.json','w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))

    y_test_pred = model.predict(X_test, batch_size=256).argmax(axis=1)
    pd.DataFrame({"pred": ['NOT' if y==0 else 'OFF' for y in y_test_pred]}).to_csv('outputs/lstm_test_predictions.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()
