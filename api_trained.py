from torch import optim
from tqdm.auto import tqdm
from helper import *
from model.generator import SkipEncoderDecoder, input_noise
import os

def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100, train=False):
    DTYPE = torch.FloatTensor
    if torch.cuda.is_available():
        device = 'cuda'
        print("Setting Device to CUDA...")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Setting Device to MPS...")
    else:
        device = 'cpu'
        print('\nSetting device to "cpu", since torch is not built with "cuda" or "mps" support...')
        print('It is recommended to use GPU if possible...')

    # Preprocess images
    image_np, mask_np = preprocess_images(image_path, mask_path, max_dim)

    depth = 192

    # Load the model
    print('Building the model...')
    generator = SkipEncoderDecoder(
        input_depth,
        num_channels_down=[depth] * 5,
        num_channels_up=[depth] * 5,
        num_channels_skip=[depth] * 5
    )

    # Load the pre-trained model
    if os.path.exists(f'generator_pretrained_{depth}.pth'):
        generator.load_state_dict(torch.load(f'generator_pretrained_{depth}.pth', map_location=torch.device(device)))
        print(f'Loaded pre-trained generator model with depth {depth}.')
    else:
        raise FileNotFoundError(f'Pre-trained model generator_pretrained_{depth}.pth not found.')

    generator.type(DTYPE).to(device)

    # Prepare inputs for inference
    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)
    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    if train:
        # Training logic
        generator_input_saved = generator_input.detach().clone()
        noise = generator_input.detach().clone()

        objective = torch.nn.MSELoss().type(DTYPE).to(device)
        optimizer = optim.Adam(generator.parameters(), lr)

        print('\nStarting training...\n')
        progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

        for step in progress_bar:
            optimizer.zero_grad()
            generator_input = generator_input_saved

            if reg_noise > 0:
                generator_input = generator_input_saved + (noise.normal_() * reg_noise)

            output = generator(generator_input)
            loss = objective(output * mask_var, image_var * mask_var)
            loss.backward()

            if step % show_step == 0:
                output_image = torch_to_np_array(output)
                # visualize_sample(image_np, output_image, nrow=2, size_factor=10)

            progress_bar.set_postfix(Loss=loss.item())
            if loss.item() < 0.00015:
                break

            optimizer.step()

        torch.save(generator.state_dict(), f'generator_pretrained_{depth}.pth')
        output_image = torch_to_np_array(output)

    else:
        # Inference logic - no training, just load and apply the model
        print('\nStarting inference...\n')
        with torch.no_grad():
            output = generator(generator_input)
            output_image = torch_to_np_array(output)

    # Save the final output image
    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))
    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')
    pil_image.save(output_path)

