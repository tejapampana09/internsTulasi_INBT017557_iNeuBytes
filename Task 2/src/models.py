import tensorflow as tf
from tensorflow.keras import layers, models, initializers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def get_logistic_regression(random_state=42):
    return LogisticRegression(max_iter=1000, random_state=random_state)

def get_svm(random_state=42):
    return LinearSVC(max_iter=2000, random_state=random_state)

def get_lstm_model(vocab_size, embedding_dim, max_length, lstm_units=64, dropout_rate=0.3, embedding_matrix=None):
    """
    Builds and compiles an LSTM model for binary sentiment classification.
    """
    model = models.Sequential()
    
    if embedding_matrix is not None:
        # Load pre-trained embeddings (static/non-trainable)
        model.add(layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            embeddings_initializer=initializers.Constant(embedding_matrix),
            trainable=False
        ))
    else:
        # Trainable embedding from scratch
        model.add(layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length
        ))
        
    model.add(layers.LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.0))
    model.add(layers.Dense(lstm_units // 2, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model
