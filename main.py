from project_torch import EEGNet
from project_torch_img import EfficientNet


model_EEGNet = True
deap = True
amigos = False
cross_subject = True
scheduler = True
original_classes = True
generate_images = False
generate_labels = False
valence = False
arousal = True
num_epochs = 200


def main():
    if model_EEGNet:
        EEGNet(num_epochs=num_epochs, valence=valence, arousal=arousal, deap=deap, amigos=amigos,
               cross_subject=cross_subject, use_scheduler=scheduler)
    else:
        EfficientNet(generate_images=generate_images, generate_labels=generate_labels,
                     original_classes=original_classes, valence=valence, arousal=arousal)


if __name__ == '__main__':
    main()
