import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

class WebClickEnv(gym.Env):
    """
    A custom Gym environment simulating a web page with clickable buttons.

    Actions:
        - Continuous: A tuple (x, y) representing the coordinates to click.

    Observation:
        - An RGB image representation of the web page with buttons.
        - The image has a shape of (300, 300, 3) and pixel values ranging from 0 to 255.

    Termination:
        - The episode ends when a button is clicked.

    Reward:
        - A reward of 1 is given if the click is within the button area.
        - A negative reward is given based on the distance to the button otherwise.

    Initialization:
        - Button position is randomized within the window boundaries.
    """
        
    def __init__(self):
        super(WebClickEnv, self).__init__()
        self.button_size = 50
        self.button_position = (0, 0)
        self.height=300
        self.width=400
        self.action_space = spaces.Box(low=0, high=max(self.height,self.width), shape=(2,), dtype=np.float32)  # (x, y) coordinates
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.width,self.height, 3), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.button_position = (random.randint(0, self.width - self.button_size), random.randint(0, self.height - self.button_size))
        return self._get_observation(), {"Button Position": self.button_position}

    def step(self, action):
        # Action normalization
        terminated = False
        truncated = False
        x, y = action
        reward = -1  # Default penalty

        if (self.button_position[0] <= x <= self.button_position[0] + self.button_size and
            self.button_position[1] <= y <= self.button_position[1] + self.button_size):
            reward = 1  # Clicked the button
        else:
            # Calculate distance penalty
            distance = np.sqrt((x - (self.button_position[0] + self.button_size / 2))**2 +
                               (y - (self.button_position[1] + self.button_size / 2))**2)
            reward = -distance / 300  # Normalized distance penalty

        return self._get_observation(), reward,terminated, truncated, {"Button Position": self.button_position}

    def _get_observation(self):
        font_size=16
        font = ImageFont.truetype('arial.ttf', font_size)
        img = Image.new("RGB", (self.width,self.height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([self.button_position[0], self.button_position[1],
                        self.button_position[0] + self.button_size,
                        self.button_position[1] + self.button_size], fill="blue")
        
            # Calculate text position (centered)
        text = "Search"
        textheight = font_size
        textwidth = draw.textlength(text, font)
        text_x = self.button_position[0] + (self.button_size - textwidth) / 2
        text_y = self.button_position[1] + (self.button_size - textheight) / 2
        # Draw the text
        draw.text((text_x, text_y), text, font=font, fill="white")

        return np.array(img)
