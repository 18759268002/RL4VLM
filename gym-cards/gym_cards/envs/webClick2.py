import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw, ImageFont

class WebClickEnv2(gym.Env):
    """
    A custom Gym environment simulating a web page with clickable buttons and news articles.

    Actions:
        - Continuous: A tuple (x, y) representing the coordinates to click.

    Observation:
        - An RGB image representation of the web page with buttons and news articles.
        - The image has a shape of (300, 400, 3) and pixel values ranging from 0 to 255.

    Termination:
        - The episode ends when a clickable title is clicked.

    Reward:
        - A reward of 1 is given if the click is within the title area.
        - A negative reward is given based on the distance to the title otherwise.

    Initialization:
        - Randomly selects a news article to be the goal.
    """

    def __init__(self):
        super(WebClickEnv2, self).__init__()
        self.button_size = 50
        self.button_position = (0, 0)
        self.height = 300
        self.width = 400
        self.action_space = spaces.Box(low=0, high=max(self.height, self.width), shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        
        # News articles with titles and bodies
        self.news_articles = [
            {"title": "Breaking News: AI Revolution", "body": "Artificial intelligence is transforming industries.","y_position":"0"},
            {"title": "Sports Update: Local Team Wins", "body": "The local team clinched the championship.","y_position":"120"},
            {"title": "Weather Alert: Storm Incoming", "body": "A storm is expected to hit the area this weekend.","y_position":"240"}
        ]
        self.articale_mode = True
        self.goal_article = None
        self.reset()

    def reset(self):
        # Randomly select one article to be the goal
        self.articale_mode = True
        self.goal_article = random.choice(self.news_articles)
        self.button_position = (random.randint(0, self.width - self.button_size), random.randint(0, self.height - self.button_size))
        return self._get_observation(), {"Goal Article": self.goal_article["title"]}

    def step(self, action):
        # Action normalization
        terminated = False
        truncated = False
        x, y = action
        reward = -1  # Default penalty

        if self.articale_mode:
            # Calculate title area based on selected goal article
            title_position = (20, self.goal_article["y_position"])  # Example position for title
            title_height = 30  # Height of the title
            title_width = 0  # Calculate width based on text size later

            # Create the image to measure the title width
            img = Image.new("RGB", (self.width, self.height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype('arial.ttf', title_height)
            title_width = draw.textlength(self.goal_article["title"], font)

            # Check if the click is within the title area
            if (title_position[0] <= x <= title_position[0] + title_width and
                title_position[1] <= y <= title_position[1] + title_height):
                reward = 1  # Clicked the title
                terminated = True
            else:
                # Calculate distance penalty
                distance = np.sqrt((x - (title_position[0] + title_width / 2))**2 +
                                (y - (title_position[1] + title_height / 2))**2)
                reward = -distance / 300  # Normalized distance penalty

            return self._get_observation(), reward, terminated, truncated, {"Goal Article": self.goal_article["title"]}

        else:
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
        if self.articale_mode:
            font_size_title = 24
            font_size_body = 16
            img = Image.new("RGB", (self.width, self.height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            # Loop through the articles
            y_position = 0
            for article in self.news_articles:
                # Draw the title
                title_font = ImageFont.truetype('arial.ttf', font_size_title)
                draw.text((20, y_position), article["title"], font=title_font, fill="black")
                y_position += 40  # Move down for the body text
                # Draw the body
                body_font = ImageFont.truetype('arial.ttf', font_size_body)
                draw.text((20, y_position), article["body"], font=body_font, fill="black")
                y_position += 80  # Move down for the next article

            return np.array(img)

        
        else:
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
