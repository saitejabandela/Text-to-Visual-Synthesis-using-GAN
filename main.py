# This is a sample Python script.
from GAN import GAN
from Generator import Generator
from Discriminator import Discriminator
from StackGanStage1 import StackGanStage1
from StackGanStage2 import StackGanStage2
from Utility import Utility
from InceptionScore import InceptionScore

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def runStackGAN():
    generator = Generator()
    generatorInfo = generator.stage1_generator()
    generatorInfo.summary()

    discriminator = Discriminator()
    discriminatorInfo = discriminator.stage1_discriminator()
    discriminatorInfo.summary()

    gan = GAN()
    ganstage1 = gan.stage1_adversarial(generator, discriminator)
    ganstage1.summary()

    stage1 = StackGanStage1()
    stage1.train_stage1()

    generator_stage2 = generator.build_stage2_generator()
    generator_stage2.summary()

    discriminator_stage2 = discriminator.stage2_discriminator()
    discriminator_stage2.summary()

    adversarial_stage2 = gan.stage2_adversarial_network(discriminator_stage2, generator_stage2, generator)
    adversarial_stage2.summary()

    stage2 = StackGanStage2()
    stage2.train_stage2()

    # Path to the folder containing your generated images
    folder_path = '/Users/tejasree/Downloads/Stage2Results50epoch'
    inception_score = InceptionScore()
    # Load images
    generated_images = inception_score.load_images_from_folder(folder_path)

    # Calculate Inception Score
    inception_score = inception_score.calculate_inception_score(generated_images)
    print("Inception Score:", inception_score)


if __name__ == '__main__':
    runStackGAN()
