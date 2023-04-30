# High Dynamic Range Imaging

## Environment & Package

`Python 3.11.2`

`OpenCV 4.7.0`

`Numpy 1.24.2`

## How to run the project
1. Put the dictionary `raw_image` and source code under the same dictionary 
2. Run `python3 main.py` in terminal (You can also define your own arguments when running `main.py`)

## Some details
- Main program `main.py`

  - Users can define arguments in `main.py` 

  ```python
  parser = argparse.ArgumentParser(description='main function of High Dynamic Range Imaging')
  parser.add_argument('--alignment', default=1, type=int, help='align image or not')
  parser.add_argument('--delta', default=0.000001, type=float, help='delta for tone mapping')
  parser.add_argument('--alpha', default=0.5, type=float, help='alpha for tone mapping')
  parser.add_argument('--L_white', default=1.3, type=float, help='L_white for tone mapping')
  parser.add_argument('--image_path', default='./raw_image/', help='path to input image')
  parser.add_argument('--noise', default=2.0, type=float, help='threshold value for removing noise in MTB alignment')
  parser.add_argument('--level', default=4, type=int, help='pyramid level in MTB alignment')
  args = parser.parse_args()
  ```

- Algorithms are implemented in other programs

  - `MTB_alignment.py`：Implement MTB alignment algorithm discussed in class
    According to this algorithm, we need to construct bit map, image pyramid, and also calculate XOR difference at each candidate offset. To speed up the algorithm, I use some functions from **opencv** and **Numpy**. 
    1. For generating image pyramid, we need to resize the image to 1/4 smaller. I use `cv2.resize` here and resize its width and height to 1/2 smaller hence generating a image with 1/4 smaller size of the origin image.
    2. For computing the XOR difference at each offset and taking the coordinate pair corresponding to the minimum difference, I use `cv2.warpAffine` to help generating shifted image for each candidate offset faster. Besides, I use `np.logical_xor` to efficiently compute the XOR difference.
  - `hdr.py`：reconstruct radiance map from raw images and assemble HDR image
    Due to the linear property of digital value function saved in RAW image, we can merely divide sensor exposure with exposure time to reconstruct the target radiance function. That is,
    $$ E_{i} = \frac{X_{ij}}{\Delta t_j}\qquad\text{(for i-th point and j-th image)} $$
  - `tone_mapping.py`：Implement global tone mapping algorithm discussed in class
    We use **Reinhard_tonemap** to implement tone mapping, so we compute luminances as following 
    $$\overline{L}_w=\exp{\left(\frac{1}{N}\sum_{x,y}\log(\delta+L_w)\right)}.$$
    Then, we turn it into 
    $$L_m=\alpha\times\frac{L_w}{\overline{L}_w}$$
    to derive displayed luminances
    $$L_d=\frac{L_m\left(1+\frac{L_m}{1.5^2}\right)}{1+L_m}$$
    In addition, we implement color correction after tone mapping since we discovered that there is a color distortion in the image we derived.
