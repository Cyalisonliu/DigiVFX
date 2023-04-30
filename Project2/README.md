# Image stitching

##  Dependency

* `Python3`
  * `OpenCV 4.7.0`
  * `Numpy 1.24.2`
  * `scikit-learn 1.0.2`
  * `scipy 1.7.3`
  * `matplotlib 3.7.1` (only for showing images generated during process)

## How to run the project to get the same result

1. Put the directory `photos` (put original photos inside)  and `input` (cylinder.py will generate images after cylinder projection here) and source code under the same dictionary 
2. Run `python3 main.py` in terminal (You can also define your own arguments when running `main.py`)

## Program Structure

<img src="/Users/alison/Desktop/Screenshot 2023-04-29 at 10.40.46 PM.png" alt="Screenshot 2023-04-29 at 10.40.46 PM" style="zoom:45%;" />

## Detailed Usage

* **main.py**

  * Main program
  * Users can define only one argument in `main.py` to see the process of our image stitching project

  ```python
  parser = argparse.ArgumentParser(description='main function of Image Stitching Project')
  parser.add_argument('--draw_process', default=0, type=int, help='Set this to 1 is you want to see the process')
  args = parser.parse_args()
  ```

### Algorithms are implemented in other programs

* **cylinder.py**

  we adopt the following formula to derive the new coordinate for every pixel.
  $$ x' = f\arctan\frac{x}{f}\quad\text{and}\quad y' = f\frac{y}{\sqrt{x^2+f^2}} $$

  Store the images to directory `input` after cylinder projection.

* **drawplot.py**

  This program help doing vitualization when we run the project.

* **sift_detector.py**
  We implement **SIFT(Scale-Invariant Feature Transform)** here to extract keypoints and descriptors for feature matching.

  ```python
  def SIFT_get_features(img, draw, s=3, num_octave=4, sigma=1.6, curvature_threshold=10.0, contrast_threshold=3.5):
  # The SIFT_get_features function is initialized with s=3, num_octave=4, sigma=1.6, curvature_threshold=10.0, contrast_threshold=3.5, where
  """
      Args:
      img - array of the input image on the LHS
      draw - show the images generated during process
      s - the layer of Difference of Gaussian generated
      num_octave = number of octave in SIFT
      sigma - the sigma in gaussian filter
      curvature_threshold - curvature threshold for removing unprecised keypoints located in egdes
      contrast_threshold - contrast threshold for removing unprecised keypoints

      Returns
      kp_pos - the x, y of all keypoints
      descriptor - descriptor for each keypoint
  """
  ```

  

* **utilis.py**
  This program define all the functions we need when implementing SIFT. The overall structure is shown below. For details, there are some descriptions in the program too.

  ```python
  """
  Define gaussain filter
  """
  def get_gaussain_filter(sigma)
  
  """
  Code for generating Gaussain pyrimid & DoG pyrimid
  """
  def generate_octave(first_img, s, sigma):
  def generate_pyramid(base_img, num_octave, s, sigma, subsample):
    
  """
  Code for detecting keypoints
  """
  def get_keypoints(DOG_pyr, num_octave, s, subsample, contrast_threshold, curvature_threshold):
    
  """
  Code for assigning orientation to keypoints
  """
  def assign_orientation(kp_pyr, gaussain_pyr, s, num_octave, subsample):
  
  """
  Code to extract feature descriptors for the keypoints.
  """
  def generate_descriptor(kp_pos, gaussain_pyr, orient, scale, subsample):
  ```

  

* **feature_matching.py**

  Here, we use the **KDTree** library from **sklearn.neighbors** to help us speed up matching. After that, we go through each pair of keypoints. And we set the distance threshold to be 2.5. Only the pair of keypoins with distance smaller than the threshold will remain.

    ```python
    """
        Args:
        img1 - array of the image on the LHS. Also seen as destination image
        kp1 - N*2 keypoints array, N is the number of keypoints of the image on the LHS we found in SIFT
        des1 - N*128 descriptors array, N is the number of keypoints of the image on the LHS we found in SIFT
        kp2 - N*2 keypoints array, N is the number of keypoints of the image on the RHS we found in SIFT 
        des2 - N*128 descriptors array, N is the number of keypoints of the image on the RHS we found in SIFT
    
        Return:
        matched_pairs: An array with the matched keypoints we found in RANSAC
    """
    ```

* **image_matching.py**

  We use a robust estimation technique RANSAC to estimate the offsets between given pair of images. Practically, we set the threshold as 2.5 pixels, and since we don’t really have a large number of feature pair to be matched, we just go
  through all the pairs. 

  ```python
  """
      Args:
      src_points - N*2 source pixel location matrices, N is the number of matched keypoints
      dst_points - N*2 destination pixel location matrices, N is the number of matched keypoints
      
      Return:
      best_offset: A list with the current offsets we found between a set of matched keypoints
  """
  ```

* **stitch.py**

    Stiching the image with all the offsets we found. we also apply the linear blending to refine the output image by dealing with overlapping problem. The result will be saved in the same directory as `result_noCrop.jpg` and `result_crop.jpg` which represent the result image before and after cropped respectively.

    ```python
    """
        Args:
        img_path - path of all the image we want to stitch together
        offsets - all the offsets we found from image matching
        
        Return:
       	result_image - result image without cropped
       	crop_img - result image after cropped
    """
    ```
