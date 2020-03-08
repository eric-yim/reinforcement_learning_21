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

<img src="https://user-images.githubusercontent.com/48815706/76157607-b92d5f80-60bf-11ea-8ce8-416971259feb.jpg">

<br>I trained the agent using the TF Agents PPO. The learned action network follow how an intelligent human player would play. The learned value net, has some oddities, but show overall that the agent expects to lose.

<img src="https://user-images.githubusercontent.com/48815706/76157610-cea28980-60bf-11ea-9d26-7c926ab91dfb.jpg">

<br>For example, when player has 19 and dealer shows a 2, the action network suggests an action of 0 (Stand) with high confidence.
<br>[19] [2] [[-0.22117655]] [[0.8581143  0.09495622 0.04692944]]
<br>When the player has 15 and the dealer shows a 10, the agent's best move is still to Stand, but it has low confidence.
<br>[15] [10] [[-2.1670158]] [[0.4792729  0.2612255  0.25950167]]
<br>Observing the value network outputs, the agent expects to lose roughly 2 hands by the end of the deck, and as the deck plays out, the agent values the situation less negative. Aside from this overall increasing trend, the agent values some observations better than others.
<br>When player has 20 and the dealer shows a 2, this is valued positively.
<br>[20] [2] [[0.6782062]] 
<br>When player has 19, but dealer shows an Ace, this is valued very negatively. Even though a 19 is a high number, the ace's flexibility to be either 1 or 11 in BlackJack is strong for the dealer. This is worsened by the fact that 10s are the most numerous in the deck.
<br>[19] [1] [[-1.744565]]
<br>Overall, I was disappointed the agent could not learn a way to consistently best the dealer even with card counting and the altered DoubleDown rule. However, I did not investigate whether it is theoretically possible to beat the dealer 1-on-1. <br>Looking at the results another way: if I had no knowledge of the actual mathematical odds, I might assume because the environment is simple, and because the PPO Agent takes a near optimal set of actions and still loses, I might conclude is it impossible to beat a dealer in this set up.
