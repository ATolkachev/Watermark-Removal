from torch import optim
from tqdm.auto import tqdm
from helper import *
from model.generator import SkipEncoderDecoder, input_noise
import os


def remove_watermark(image_path, mask_path, max_dim, reg_noise, input_depth, lr, show_step, training_steps, tqdm_length=100, early_stopping_threshold=0.00015, validation_frequency=100):
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

    # Load the pre-trained model if available
    if os.path.exists(f'generator_pretrained_{depth}.pth'):
        generator.load_state_dict(torch.load(f'generator_pretrained_{depth}.pth', map_location=torch.device(device)))
        print(f'Loaded pre-trained generator model with depth {depth}.')

    generator.type(DTYPE).to(device)

    # Define loss function and optimizer
    objective = torch.nn.MSELoss().type(DTYPE).to(device)
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    # Prepare inputs for training
    image_var = np_to_torch_array(image_np).type(DTYPE).to(device)
    mask_var = np_to_torch_array(mask_np).type(DTYPE).to(device)
    generator_input = input_noise(input_depth, image_np.shape[1:]).type(DTYPE).to(device)

    generator_input_saved = generator_input.detach().clone()
    noise = generator_input.detach().clone()

    best_val_loss = float('inf')

    print('\nStarting training...\n')
    progress_bar = tqdm(range(training_steps), desc='Completed', ncols=tqdm_length)

    for step in progress_bar:
        generator.train()
        optimizer.zero_grad()
        generator_input = generator_input_saved

        if reg_noise > 0:
            generator_input = generator_input_saved + (noise.normal_() * reg_noise)

        # Forward pass
        output = generator(generator_input)

        # Compute loss
        loss = objective(output * mask_var, image_var * mask_var)
        loss.backward()

        # Update model weights
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix(Loss=loss.item())

        # Early stopping based on loss threshold
        if loss.item() < early_stopping_threshold:
            print(f"Early stopping as loss has reached below {early_stopping_threshold}")
            break

        # Validation step
        if step % validation_frequency == 0:
            generator.eval()
            with torch.no_grad():
                val_output = generator(generator_input)
                val_loss = objective(val_output * mask_var, image_var * mask_var)
                print(f'Step {step}, Validation Loss: {val_loss.item()}')

                # Save the best model
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(generator.state_dict(), 'best_generator_model.pth')

    # Save final model
    torch.save(generator.state_dict(), f'generator_pretrained_{depth}.pth')

    # Generate final output image
    generator.eval()
    with torch.no_grad():
        output = generator(generator_input)
    output_image = torch_to_np_array(output)

    # Save the final output image
    pil_image = Image.fromarray((output_image.transpose(1, 2, 0) * 255.0).astype('uint8'))
    output_path = image_path.split('/')[-1].split('.')[-2] + '-output.jpg'
    print(f'\nSaving final output image to: "{output_path}"\n')
    pil_image.save(output_path)