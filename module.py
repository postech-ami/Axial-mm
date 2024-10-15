# References
# 1. https://people.csail.mit.edu/tiam/deepmag/

import torch
import torch.nn as nn
  
# modules
class res_blk(nn.Module):
  def __init__(self, layer_dims, ks, s, p):
    super(res_blk, self).__init__()
    self.conv1 = nn.Conv2d(layer_dims, layer_dims, kernel_size=ks, stride=s, padding=p, padding_mode='reflect')
    self.conv2 = nn.Conv2d(layer_dims, layer_dims, kernel_size=ks, stride=s, padding=p, padding_mode='reflect')
    self.activation = nn.ReLU()

  def forward(self, input):
    out = self.conv1(input)
    out = self.activation(out)
    out = self.conv2(out)
    return input + out

# Residual blocks
def multi_res_blk(num_res_blk, layer_dims, ks, s, p):
  layers = []
  # num_res_blk=3. layer_dims=32, ks=3, s=1
  for i in range(num_res_blk):
    layers.append(res_blk(layer_dims, ks, s, p))
  return nn.Sequential(*layers)

# Manipulator
class res_manipulator(nn.Module):
  def __init__(self, args, device, layer_dims=32):
    super(res_manipulator, self).__init__()
    
    ld1, ld2, ld3, ld4 = 32, 32, 32, 32
    
    self.conv1 = nn.Conv2d(ld1, ld2, kernel_size=[7, 1], stride=1, padding=[3, 0], padding_mode='reflect')
    self.conv2 = nn.Conv2d(ld3, ld4, kernel_size=[3, 1], stride=1, padding=[1, 0], padding_mode='reflect')

    self.residual = multi_res_blk(1, layer_dims, [3, 1], 1, [1, 0])
    self.activation = nn.ReLU()
    self.args = args

  def forward(self, prev_x, prev_y, next_x, next_y, mag_map, theta, flow=False):

    # Projection Layer: X&Y axes shape representations -> Axial shape representations
    prev_theta = prev_x * torch.cos(theta) - prev_y * torch.sin(theta)
    prev_theta_ortho = prev_x * torch.sin(theta) + prev_y * torch.cos(theta)

    next_theta = next_x * torch.cos(theta) - next_y * torch.sin(theta)
    next_theta_ortho = next_x * torch.sin(theta) + next_y * torch.cos(theta)

    # Obtain the differences of axial shape representations
    diff_theta = next_theta - prev_theta
    diff_theta_ortho = next_theta_ortho - prev_theta_ortho

    # Magnify the differences of axial shape representations
    diff_theta *= mag_map[:, :1, ...]
    diff_theta_ortho *= mag_map[:, 1:, ...]

    diff_theta_ortho = diff_theta_ortho.permute(0, 1, -1, -2)

    # Apply the same 1D convolution, but in difference directions
    diff_theta = self.activation(self.conv1(diff_theta))
    diff_theta_ortho = self.activation(self.conv1(diff_theta_ortho))

    diff_theta = self.conv2(diff_theta)
    diff_theta = self.residual(diff_theta)

    diff_theta_ortho = self.conv2(diff_theta_ortho)
    diff_theta_ortho = self.residual(diff_theta_ortho)

    diff_theta_ortho = diff_theta_ortho.permute(0, 1, -1, -2)

    # Inverse projection Layer: Axial shape representations -> X&Y axes shape representations
    diff_x = diff_theta * torch.cos(theta) + diff_theta_ortho * torch.sin(theta)
    diff_y = -diff_theta * torch.sin(theta) + diff_theta_ortho * torch.cos(theta)

    return next_x + diff_x, next_y + diff_y

# Encoder before tex, shape encoder
class res_encoder(nn.Module):
  def __init__(self, layer_dims=32, num_res_blk=3):
    super(res_encoder, self).__init__()
    self.conv1 = nn.Conv2d(3, int(layer_dims / 2), kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect')
    self.conv2 = nn.Conv2d(int(layer_dims / 2), layer_dims, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect')
    self.residual = multi_res_blk(num_res_blk, layer_dims, 3, 1, 1)
    self.activation = nn.ReLU()

  def forward(self, x):
    out = self.activation(self.conv1(x))
    out = self.activation(self.conv2(out))
    out = self.residual(out)
    return out

# Decoder part after upsampling
class res_decoder(nn.Module):
  def __init__(self, layer_dims=64, num_res_blk=9):
    super(res_decoder, self).__init__()
    self.residual = multi_res_blk(num_res_blk, layer_dims, 3, 1, 1)
    self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest')
    self.conv_up_sample= nn.Conv2d(layer_dims,layer_dims,kernel_size=3,stride=1,padding=1,padding_mode = 'reflect')
    self.conv1 = nn.Conv2d(layer_dims, int(layer_dims / 2), kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect') ## change
    self.conv2 = nn.Conv2d(int(layer_dims / 2), 3, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect') ## change
    self.activation = nn.ReLU()

  def forward(self, x):
    out = self.residual(x)
    out = self.up_sample(out)
    out = self.activation(self.conv_up_sample(out))
    out = self.activation(self.conv1(out))
    out = self.conv2(out)
    return out

# magnet
class encoder(nn.Module):
  def __init__(self):
    super(encoder, self).__init__()
    # set variables
    self.num_enc_resblk = 3
    self.res_enc_dim = 32
    self.num_texture_resblk = 2
    self.num_shape_resblk = 3

    # set arch
    self.res_encoder = res_encoder(self.res_enc_dim ,self.num_enc_resblk)
    self.conv_tex = nn.Conv2d(self.res_enc_dim, self.res_enc_dim, kernel_size = 3, stride = 2, padding = 1, padding_mode = 'reflect') # stride is 2, cause texture_downsample is True, else 1
    self.conv_sha = nn.Conv2d(self.res_enc_dim, self.res_enc_dim, kernel_size = [3, 1], stride = 1, padding = [1, 0], padding_mode = 'reflect')
    self.texture_resblk = multi_res_blk(self.num_texture_resblk, self.res_enc_dim, 3, 1, 1)
    self.shape_resblk = multi_res_blk(self.num_shape_resblk, self.res_enc_dim, [3, 1], 1, [1, 0])

    self.activation = nn.ReLU()

  def forward(self, img):
    enc = self.res_encoder(img)

    texture_enc = enc

    # extract texture output
    texture_enc = self.activation(self.conv_tex(texture_enc))
    texture_enc = self.texture_resblk(texture_enc)

    shape_enc_x = enc
    shape_enc_y = enc.permute(0, 1, -1, -2)

    # extract shape_x output
    shape_enc_x = self.activation(self.conv_sha(shape_enc_x))
    shape_enc_x = self.shape_resblk(shape_enc_x)
    
    # extract shape_y output
    shape_enc_y = self.activation(self.conv_sha(shape_enc_y))
    shape_enc_y = self.shape_resblk(shape_enc_y)
    shape_enc_y = shape_enc_y.permute(0, 1, -1, -2)

    return texture_enc, shape_enc_x, shape_enc_y

class decoder(nn.Module):
  def __init__(self):
    super(decoder, self).__init__()
    # set variables
    self.num_dec_resblk = 10
    self.texture_dims = 32
    self.shape_dims = 32
    self.decoder_dims = self.texture_dims + self.shape_dims

    self.conv1 = nn.Conv2d(self.shape_dims*2, self.shape_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
    self.conv2 = nn.Conv2d(self.shape_dims, self.shape_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
    self.activation = nn.ReLU()

    # set arch
    self.up_sample = nn.Upsample(scale_factor = 2, mode = 'nearest') # when texture representation downsample, activate it
    self.conv_tex_aft_upsample = nn.Conv2d(self.texture_dims, self.texture_dims, kernel_size = 3, stride = 1, padding = 1, padding_mode = 'reflect')
    self.res_decoder = res_decoder(self.decoder_dims, self.num_dec_resblk)
    self.activation = nn.ReLU()

  def forward(self, texture_enc, shape_enc_x, shape_enc_y):

    shape_enc = torch.cat([shape_enc_x, shape_enc_y], dim=1)
    shape_enc = self.conv2(self.activation(self.conv1(shape_enc)))

    texture_enc = self.up_sample(texture_enc) # when texture representation downsample, activate it
    texture_enc = self.activation(self.conv_tex_aft_upsample(texture_enc))

    enc = torch.cat((texture_enc, shape_enc), 1)
    return self.res_decoder(enc)

class magnet(nn.Module):
  def __init__(self, args, device):
    super(magnet, self).__init__()
    self.encoder = encoder()
    self.decoder = decoder()
    self.res_manipulator = res_manipulator(args, device)

  def forward(self, amplified, image_a, image_b, image_c, mag_map, theta_map):
    texture_amp, _, _ = self.encoder(amplified)
    texture_a, shape_ax, shape_ay = self.encoder(image_a)
    texture_b, shape_bx, shape_by = self.encoder(image_b)
    texture_c, shape_cx, shape_cy = self.encoder(image_c)
    out_shape_enc_x,out_shape_enc_y  = self.res_manipulator(shape_ax, shape_ay, shape_bx, shape_by, mag_map, theta_map)
    out = self.decoder(texture_b, out_shape_enc_x, out_shape_enc_y) 

    return out, texture_a, texture_c, texture_b, texture_amp, shape_bx, shape_by, shape_cx, shape_cy
  
  def inference(self, image_a, image_b, mag_map, theta_map):
    texture_a, shape_ax, shape_ay = self.encoder(image_a)
    texture_b, shape_bx, shape_by = self.encoder(image_b)
    out_shape_enc_x,out_shape_enc_y  = self.res_manipulator(shape_ax, shape_ay, shape_bx, shape_by, mag_map, theta_map)
    out = self.decoder(texture_b, out_shape_enc_x, out_shape_enc_y) 

    return out