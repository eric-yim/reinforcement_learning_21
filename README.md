# Reinforcement Learning for BlackJack
<br>Using Tensorflow 2.0 / TF Agents
<br>Using PPO

<br>Here, I create a BlackJack game (the environment), allow the agent to play through many decks, and let the agent optimize its actions (with PPO).
The environment is set as follows:
<ul>
<li>One scene is one deck of 52 cards</li>
<li>The observations are the player's cards, one of the dealer's cards, and all previous cards that have shown (because I was hoping the agent would learn to count cards).</li>
<li>Possible actions are 0)Stand 1)Hit 2)DoubleDown</li>
<li>Rewards are +1 if WIN, -1 if LOSE</li>
<li>If DoubleDown chosen, player gets a single card and the reward is doubled.</li>
</ul>
<br>Note because the player acts before the dealer, and if the player goes over 21, he automatically loses without the dealer needing to draw cards, the dealer is greatly favored in BlackJack. In the actual game, the player can only DoubleDown if he has 2 cards, but in this simulated environment, the agent can DoubleDown on any turn.
