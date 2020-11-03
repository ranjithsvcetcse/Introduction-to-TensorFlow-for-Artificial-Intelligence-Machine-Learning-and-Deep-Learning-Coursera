# Introduction-to-TensorFlow-for-Artificial-Intelligence-Machine-Learning-and-Deep-Learning-Coursera
This repository contains the assignments for the Coursera course Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning.

Check the Quiz Answers [here...](https://www.youtube.com/playlist?list=PLMtSwsZ75jcgr_eHo1LtbhS8O0AFcroBC)

> Either find answers in above files or just copy from below codes..



# Exercise 1 - House Prices Question


```Python
# GRADED FUNCTION: house_model

def house_model(y_new):

    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)

    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    model.compile(optimizer='sgd', loss='mean_squared_error')

    model.fit(xs, ys, epochs=500)

    return model.predict(y_new)[0]
```
# Exercise 2

```python
# GRADED FUNCTION: train_mnist

def train_mnist():

    # Please write your code only where you are indicated.

    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            if(logs.get('acc')>0.99):

              print("\nReached 99% accuracy so cancelling training!")

              self.model.stop_training = True

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    # YOUR CODE SHOULD START HERE

    x_train, x_test = x_train / 255.0, x_test / 255.0

    callbacks = myCallback()

    # YOUR CODE SHOULD END HERE

    model = tf.keras.models.Sequential([

        # YOUR CODE SHOULD START HERE

      tf.keras.layers.Flatten(input_shape=(28, 28)),

      tf.keras.layers.Dense(512, activation=tf.nn.relu),

      tf.keras.layers.Dense(10, activation=tf.nn.softmax)

        # YOUR CODE SHOULD END HERE

    ])

    model.compile(optimizer='adam',

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    

    # model fitting

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

    # model fitting

    return history.epoch, history.history['acc'][-1]
```

# Exercise 3

```python
GRADED FUNCTION: train_mnist_conv

def train_mnist_conv():

    # Please write your code only where you are indicated.

    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE

    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            if(logs.get('acc')>0.998):

                print("\nReached 99.8% accuracy so cancelling training!")

                self.model.stop_training = True

    

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist

    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    # YOUR CODE STARTS HERE

    callbacks = myCallback()

    

    training_images=training_images.reshape(60000, 28, 28, 1)

    training_images=training_images / 255.0

    test_images = test_images.reshape(10000, 28, 28, 1)

    test_images=test_images/255.0

    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([

            # YOUR CODE STARTS HERE

            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation='relu'),

            tf.keras.layers.Dense(10, activation='softmax')

            # YOUR CODE ENDS HERE

    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # model fitting

    history = model.fit(

        # YOUR CODE STARTS HERE

            training_images, training_labels, epochs=20, callbacks=[callbacks]

        # YOUR CODE ENDS HERE

    )

    # model fitting

    return history.epoch, history.history['acc'][-1]
```
# Exercise 4 Training happy sad model
```python
def train_happy_sad_model():

    # Please write your code only where you are indicated.

    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs={}):

            if(logs.get('acc')>DESIRED_ACCURACY):

                print("\nReached 99.9% accuracy so cancelling training!")

                self.model.stop_training = True

    callbacks = myCallback()

    

    # This Code Block should Define and Compile the Model

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')

    tf.keras.layers.Dense(1, activation='sigmoid')

    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.001),

              metrics=['acc'])

        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 

    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)

    train_generator = train_datagen.flow_from_directory(

        '/tmp/h-or-s/',  # This is the source directory for training images

        target_size=(150, 150),  # All images will be resized to 150x150

        batch_size=10,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')

        # Your Code Here)

    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for

    # a number of epochs.

    # model fitting

    history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,  

      epochs=15,

      verbose=1, callbacks=[callbacks])

        # Your Code Here)

    # model fitting

    return history.history['acc'][-1]
```
