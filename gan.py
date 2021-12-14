import os
import tensorflow as tf
from tqdm.auto import trange


generator, generator_optimizer = None, None
discriminator, discriminator_optimizer = None, None
noise = None
gan_checkpoint, gan_checkpoint_prefix = None, None
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

class Generator(tf.keras.models.Sequential):
    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(tf.keras.layers.Dense(5*5*256, use_bias=False, input_shape=(100,)))
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Reshape((5, 5, 256)))

        self.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert self.output_shape == (None, 5, 5, 128)
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 2), padding='same', use_bias=False))
        assert self.output_shape == (None, 5, 10, 64)
        self.add(tf.keras.layers.BatchNormalization())
        self.add(tf.keras.layers.LeakyReLU())

        self.add(tf.keras.layers.Conv2DTranspose(8, (5, 5), strides=(1, 2), padding='same', use_bias=False, activation='tanh'))
        assert self.output_shape == (None, 5, 20, 8)


class Discriminator(tf.keras.models.Sequential):
    def __init__(self):
        super().__init__()

    def create_model(self):
        self.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[5, 20, 8]))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        self.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.add(tf.keras.layers.LeakyReLU())
        self.add(tf.keras.layers.Dropout(0.3))

        self.add(tf.keras.layers.Flatten())
        self.add(tf.keras.layers.Dense(1))


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def init_gan(gan_checkpoint_dir, batch_size, noise_dim):
    global generator, generator_optimizer
    generator = Generator()
    generator.create_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    global discriminator, discriminator_optimizer
    discriminator = Discriminator()
    discriminator.create_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    global gan_checkpoint, gan_checkpoint_prefix
    gan_checkpoint_prefix = os.path.join(gan_checkpoint_dir, "ckpt")
    gan_checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    global noise
    noise = tf.random.normal([batch_size, noise_dim])


def train_gan(dataset, epochs):
    for epoch in trange(epochs, desc='train GAN'):

        for data in dataset:
            x, y = data
            image_batch = x
            train_step(image_batch)

    gan_checkpoint.save(file_prefix=gan_checkpoint_prefix)

    return generator, discriminator
