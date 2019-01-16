import numpy as np
import cv2
import time
from primesense import openni2
from primesense import _openni2 as c_api

# OPENNI_REDIST_DIR = "/home/evangeloit/Desktop/OpenNI-Linux-x64-2.2/Redist"

def write_files(dev):
    """
    Captures the point cloud and write it on a Oni file.
    """
    # user = raw_input("Press 'r' to START recording ...")
    print(" Quit :'q' , Record Frames(png) : 'p', Record (oni): 'o', Stop Recording: 's' ")

    # if user == 'r':
    openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
    depth_stream = dev.create_depth_stream()
    color_stream = dev.create_color_stream()
    print(dev.get_sensor_info(openni2.SENSOR_DEPTH))

    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                   resolutionX=640,
                                                   resolutionY=480,
                                                   fps=30))
    color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                   resolutionX=640,
                                                   resolutionY=480,
                                                   fps=30))
    depth_stream.start()
    color_stream.start()
    dev.set_image_registration_mode(True)


    shot_idx = 0
    is_png = False
    is_oni = False

    while True:
        frame_depth = depth_stream.read_frame()
        frame_color = color_stream.read_frame()

        frame_depth_data = frame_depth.get_buffer_as_uint16()
        frame_color_data = frame_color.get_buffer_as_uint8()

        depth_array = np.ndarray((frame_depth.height, frame_depth.width), dtype=np.uint16, buffer=frame_depth_data)
        color_array = np.ndarray((frame_color.height, frame_color.width, 3), dtype=np.uint8, buffer=frame_color_data)
        color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)

        cv2.imshow('Depth', depth_array)
        cv2.imshow('Color', color_array)

        ch = 0xFF & cv2.waitKey(1)

        if ch == ord('p'):
            print("type: png")
            is_png = True

        if ch == ord('o'):
            print("type: oni")
            rec = openni2.Recorder(time.strftime("%Y%m%d%H%M") + ".oni")
            rec.attach(depth_stream)
            rec.attach(color_stream)
            print(rec.start())

        if is_png == True:
            fn_depth = 'images/mydata_depth_%03d.png' % shot_idx
            fn_color = 'images/mydata_color_%03d.png' % shot_idx
            cv2.imwrite(fn_depth, depth_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(fn_color, color_array, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # print(fn_depth, 'saved')
            # print(fn_color, 'saved')
            shot_idx += 1

        if ch == ord('s'):
            is_png = False

        if ch == ord('q'):
            print("exiting...")
            depth_stream.stop()
            color_stream.stop()
            rec.stop()
            break


def main():
    """The entry point"""
    try:
        openni2.initialize("/home/evangeloit/Desktop/OpenNI-Linux-x64-2.2/Redist")  # can also accept the path of the OpenNI redistribution
    except:
        print("Device not initialized")
        return
    try:
        dev = openni2.Device.open_any()
        write_files(dev)
    except:
        print("Unable to open the device")
    try:
        openni2.unload()
    except:
        print("Device not unloaded")


if __name__ == '__main__':
    main()