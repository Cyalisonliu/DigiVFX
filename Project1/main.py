import numpy as np
import cv2
import argparse
import MTB_alignment
import tone_mapping
import hdr

def main():
    parser = argparse.ArgumentParser(description='main function of High Dynamic Range Imaging')
    parser.add_argument('--alignment', default=1, type=int, help='align image or not')
    parser.add_argument('--delta', default=0.000001, type=float, help='delta for tone mapping')
    parser.add_argument('--alpha', default=0.5, type=float, help='alpha for tone mapping')
    parser.add_argument('--L_white', default=1.3, type=float, help='L_white for tone mapping')
    parser.add_argument('--image_path', default='./raw_image/', help='path to input image')
    parser.add_argument('--noise', default=2.0, type=float, help='threshold value for removing noise in MTB alignment')
    parser.add_argument('--level', default=4, type=int, help='pyramid level in MTB alignment')
    args = parser.parse_args()
    print(args)

    # dataset of all image
    files = [ f'_DSC{idx}.ppm' for idx in range(6040, 6071)]
    expose_times = [1/4000, 1/3200, 1/2500, 1/2000, 1/1600, 1/1250, 1/1000,
                    1/800, 1/640, 1/500, 1/400, 1/320, 1/250, 1/200, 1/160,
                    1/125, 1/100, 1/80, 1/60, 1/50, 1/40, 1/30, 1/25, 1/20,
                    1/15, 1/13, 1/10, 1/8, 1/6, 1/5, 1/4]
    
    choice = [0, 5, 8, 10, 12, 14, 16, 18, 21, 26]
    print(f"Totoal number of images chosen: {len(choice)}")
    files = [files[i] for i in choice]
    expose_times = [expose_times[i] for i in choice]

    raw_images = []
    for f in files:
        print(f"reading image {f}....")
        img = cv2.imread(f"{args.image_path}{f}").astype(np.float32)
        raw_images.append(img)
    raw_images = np.array(raw_images, dtype=float)

    if args.alignment:
        # alignment
        print("processing alignment...")
        standard_idx = int(len(raw_images)/2)
        print(f"Choose {standard_idx+1} th image as referrence image")
        offsets = MTB_alignment.MTB(raw_images, standard_idx, args.level)
        aligned_images = []
        for i, img in enumerate(raw_images):
            offset = offsets[i]
            M = np.float32([ [1,0,offset[0]], [0,1,offset[1]] ])
            aligned_image = cv2.warpAffine(np.uint8(img), M, (img.shape[1],img.shape[0]))
            aligned_images.append(aligned_image)
        aligned_images = np.array(aligned_images, dtype=float)
        standard_color_img = aligned_images[-2]
         
        # reconstruct hdr image
        hdr_img = hdr.map_to_hdr(aligned_images, expose_times)
        cv2.imwrite("./recovered_HDR_image.hdr", hdr_img)

        # tone mapping
        tone_img = tone_mapping.Reinhard_tonemap(hdr_img, args.delta, args.alpha, args.L_white, standard_color_img)
        cv2.imwrite("./tonemapped_image.png", tone_img)

        # print("Tonemaping using opencv's method ... ")
        # ldrDrago = cv2.createTonemapDrago(1.2, 1.0, 0.7)
        # ldrDrago = 3 * ldrDrago.process(np.float32(hdr_img))
        # ldrDrago = cv2.cvtColor(ldrDrago, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("./ldr_Drago.jpg", ldrDrago * 255)

    else:
        hdr_img = hdr.map_to_hdr(raw_images, expose_times)
        cv2.imwrite("./noAlign_image.hdr", hdr_img)
        tone_img = tone_mapping.Reinhard_tonemap(hdr_img, args.delta, args.alpha, args.L_white)
        cv2.imwrite("./noAlign_tone_image.jpg", tone_img)

if __name__ == '__main__':
    main()