import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('./PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

client = carla.Client('localhost', 2000)
client.set_timeout(2.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
bp = blueprint_library.filter('vehicle.*')[4]
bp.set_attribute('role_name','ego')
world = client.get_world()
spawnPoint=carla.Transform(carla.Location(x=38.6,y=5.8, z=0.598),carla.Rotation(pitch=0.0, yaw=0.0, roll=0.000000))
vehicle = world.spawn_actor(bp, spawnPoint)
cam_bp = None
cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
cam_bp.set_attribute("image_size_x",str(180))
cam_bp.set_attribute("image_size_y",str(180))
cam_bp.set_attribute("fov",str(105))
cam_location = carla.Location(2,0,2)
cam_rotation = carla.Rotation(0,0,0)
cam_transform = carla.Transform(cam_location,cam_rotation)
ego_cam = world.spawn_actor(cam_bp,cam_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
#ego_cam.listen(lambda image: image.save_to_disk('tutorial/output/%.6d.jpg' % image.frame))
gnss_bp = world.get_blueprint_library().find('sensor.other.gnss')
gnss_location = carla.Location(1,0,2)
gnss_rotation = carla.Rotation(0,0,0)
gnss_transform = carla.Transform(gnss_location,gnss_rotation)
gnss = world.spawn_actor(gnss_bp,gnss_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
#Attaching Lidar
lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
lidar_location = carla.Location(0,0,2.4)
lidar_rotation = carla.Rotation(0,0,0)
lidar_transform = carla.Transform(lidar_location,lidar_rotation)
lidar =world.spawn_actor(lidar_bp,lidar_transform,attach_to=vehicle, attachment_type=carla.AttachmentType.Rigid)
time.sleep(40)
vehicle.destroy()