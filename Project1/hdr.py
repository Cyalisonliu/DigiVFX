import numpy as np

def map_to_hdr(imgs, expose_times):
    print("converting hdr image...")
    radiance_map = np.array([np.divide(imgs[i], expose_times[i]) for i in range(len(expose_times))])

    radiance_r = np.average(radiance_map[:,:,:,2], axis=0) * sum(expose_times)
    radiance_g = np.average(radiance_map[:,:,:,1], axis=0) * sum(expose_times)
    radiance_b = np.average(radiance_map[:,:,:,0], axis=0) * sum(expose_times)

    hdr = np.zeros((radiance_r.shape[0], radiance_r.shape[1], 3))
    hdr[:,:,0] = radiance_r
    hdr[:,:,1] = radiance_g
    hdr[:,:,2] = radiance_b

    return hdr