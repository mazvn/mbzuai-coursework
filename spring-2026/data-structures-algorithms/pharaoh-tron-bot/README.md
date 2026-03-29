# pharaoh - tron bot

**task:** write a bot for a multiplayer tron / snake arena. each turn the bot gets the board and has a fraction of a second to return a move. there are treasures (coins, diamonds) and abilities (boost, phase, EMP). bots play against each other and on solo maps.

**what i did:** no deep search, just simulate every legal move one step and score the result. scoring is mostly flood fill (how much space i still have after the move), number of exits, how close opponents are, and treasure value (less if an opponent is closer to it). abilities only get used when they actually improve the position. there's always an emergency fallback move ready in case time runs out. some maps get small adjustments.

only my bot is here, the game engine and the other bots were given by the course.
