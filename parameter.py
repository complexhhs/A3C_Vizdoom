# class AgentConfig(object):
# Train parameters
load_model = False

random_start = 30
discount_factor = 0.99
learn_speed = 1.5e-5
gradient_clip = 60.

# class EnvironmentConfig(object):
action_size = 3

scenario_path = "./scenarios/defend_the_center.cfg"
# scenario_path = "./scenarios/health_gathering.cfg"
map = "map01"
render_hud = True
render_crosshair = False
render_weapon = True
effect_sprites = True
render_decals = True
render_particles = True
render_messages = True
labels_buffer = True
render_corpses = True
render_screen_flash = True
time_out = 720
episode_start = 5
max_episode = 1000
window_visible = False
sound_enable = False
# living_reward = 0
living_reward = 1
# mode = Mode.PLAYER

width_screen = 84
height_screen = 84
screen_size = width_screen*height_screen

save_term = 100
step_period = 9

# model_path = './save_model_health_gather'
model_path = './save_model_defend_center'
# summary_path ='./summary_health_gather'
summary_path = './summary_defend_center'
# frame_path ='./frames_health_gather'
frame_path = './frames_defend_center'