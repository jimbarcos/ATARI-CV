import cv2
import mediapipe as mp
import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()

# Set up the game window
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Enhanced Hand-Controlled Atari Breakout")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Paddle
paddle_width, paddle_height = 100, 20
paddle = pygame.Rect(width // 2 - paddle_width // 2, height - 40, paddle_width, paddle_height)

# Ball
ball_size = 15
# Increased initial ball speed
ball_speed_x = 8 * random.choice((1, -1))
ball_speed_y = -8
ball = pygame.Rect(width // 2 - ball_size // 2, height // 2 - ball_size // 2, ball_size, ball_size)

# Bricks
brick_width, brick_height = 80, 30
bricks = [pygame.Rect(col * brick_width, row * brick_height + 50, brick_width - 2, brick_height - 2)
          for row in range(5) for col in range(width // brick_width)]

# New: Brick colors and strengths
brick_colors = [RED, BLUE, GREEN, YELLOW, WHITE]
brick_strengths = [random.randint(1, 3) for _ in range(len(bricks))]

# Game variables
score = 0
font = pygame.font.Font(None, 36)

# New: Power-ups
power_ups = []
POWER_UP_SPEED = 2
POWER_UP_TYPES = ['expand', 'shrink', 'fast', 'slow']

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def reset_game():
    global paddle, ball, bricks, brick_strengths, ball_speed_x, ball_speed_y, score, power_ups
    paddle = pygame.Rect(width // 2 - paddle_width // 2, height - 40, paddle_width, paddle_height)
    ball = pygame.Rect(width // 2 - ball_size // 2, height // 2 - ball_size // 2, ball_size, ball_size)
    ball_speed_x = 8 * random.choice((1, -1))
    ball_speed_y = -8
    bricks = [pygame.Rect(col * brick_width, row * brick_height + 50, brick_width - 2, brick_height - 2)
              for row in range(5) for col in range(width // brick_width)]
    brick_strengths = [random.randint(1, 3) for _ in range(len(bricks))]
    score = 0
    power_ups = []

# New: Function to create power-up
def create_power_up(x, y):
    power_up_type = random.choice(POWER_UP_TYPES)
    power_ups.append({'rect': pygame.Rect(x, y, 20, 20), 'type': power_up_type})

# New: Function to apply power-up effect
def apply_power_up(power_up_type):
    global paddle_width, ball_speed_x, ball_speed_y
    if power_up_type == 'expand':
        paddle_width = min(paddle_width + 20, 200)
    elif power_up_type == 'shrink':
        paddle_width = max(paddle_width - 20, 60)
    elif power_up_type == 'fast':
        ball_speed_x *= 1.2
        ball_speed_y *= 1.2
    elif power_up_type == 'slow':
        ball_speed_x *= 0.8
        ball_speed_y *= 0.8
    paddle.width = paddle_width

def game_loop():
    global ball_speed_x, ball_speed_y, score, paddle_width

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                paddle.centerx = x
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Move ball
        ball.x += ball_speed_x
        ball.y += ball_speed_y

        # Ball collision with walls
        if ball.left <= 0 or ball.right >= width:
            ball_speed_x *= -1
        if ball.top <= 0:
            ball_speed_y *= -1

        # Ball collision with paddle
        if ball.colliderect(paddle) and ball_speed_y > 0:
            ball_speed_y *= -1
            relative_intersect_x = (paddle.x + paddle.width / 2) - (ball.x + ball_size / 2)
            normalized_relative_intersect_x = relative_intersect_x / (paddle.width / 2)
            ball_speed_x = normalized_relative_intersect_x * 10  # Increased max speed

        # Ball collision with bricks
        for i, brick in enumerate(bricks[:]):
            if ball.colliderect(brick):
                brick_strengths[i] -= 1
                if brick_strengths[i] <= 0:
                    bricks.remove(brick)
                    brick_strengths.pop(i)
                    score += 10
                    # New: Chance to create a power-up
                    if random.random() < 0.2:  # 20% chance
                        create_power_up(brick.x + brick.width / 2, brick.y + brick.height)
                ball_speed_y *= -1
                # New: Increase ball speed slightly with each hit
                ball_speed_x *= 1.10
                ball_speed_y *= 1.10
                break

        # New: Update and check power-ups
        for power_up in power_ups[:]:
            power_up['rect'].y += POWER_UP_SPEED
            if power_up['rect'].colliderect(paddle):
                apply_power_up(power_up['type'])
                power_ups.remove(power_up)
            elif power_up['rect'].top >= height:
                power_ups.remove(power_up)

        # Check for game over
        if ball.bottom >= height:
            return True

        # Draw everything
        window.fill(BLACK)
        pygame.draw.rect(window, BLUE, paddle)
        pygame.draw.ellipse(window, WHITE, ball)
        for i, brick in enumerate(bricks):
            pygame.draw.rect(window, brick_colors[brick_strengths[i] - 1], brick)
        
        # New: Draw power-ups
        for power_up in power_ups:
            pygame.draw.rect(window, GREEN, power_up['rect'])

        # Draw score
        score_text = font.render(f"Score: {score}", True, WHITE)
        window.blit(score_text, (10, 10))

        pygame.display.flip()

        # Display the resulting frame
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

        clock.tick(60)

    return False

def main():
    play_again = True
    while play_again:
        reset_game()
        game_over = game_loop()
        
        if game_over:
            window.fill(BLACK)
            game_over_text = font.render("Game Over", True, WHITE)
            final_score_text = font.render(f"Final Score: {score}", True, WHITE)
            play_again_text = font.render("Press SPACE to play again or ESC to quit", True, WHITE)
            window.blit(game_over_text, (width // 2 - game_over_text.get_width() // 2, height // 2 - 50))
            window.blit(final_score_text, (width // 2 - final_score_text.get_width() // 2, height // 2 + 50))
            window.blit(play_again_text, (width // 2 - play_again_text.get_width() // 2, height // 2 + 100))
            pygame.display.flip()

            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        play_again = False
                        waiting = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            play_again = False
                            waiting = False

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()