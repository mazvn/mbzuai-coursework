# Pharaoh Tron Bot

**Task:** Write a bot for a multiplayer Tron arena. Each turn the bot gets the board and has a fraction of a second to return a move. There are treasures (coins and diamonds) and abilities (boost, phase and EMP), and bots play against each other and on solo maps.

**What I did:** No deep search. The bot simulates every legal move one step ahead and scores the result. The score is mostly flood fill (how much space is left after the move), number of exits, distance to opponents and treasure value, which is lowered when an opponent is closer to the treasure. Abilities are only used when they improve the position, and there is always a fallback move ready in case time runs out.

Only my bot is here. The game engine and the other bots were provided by the course.
