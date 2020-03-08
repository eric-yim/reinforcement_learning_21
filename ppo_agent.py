from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging

import gin
import tensorflow as tf

from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_policy
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import tensor_normalizer
from tf_agents.utils import value_ops
from tensorflow.keras import losses

from baselines.common.distributions import make_pdtype
from gym import spaces 
PPOLossInfo = collections.namedtuple('PPOLossInfo', (
    'policy_gradient_loss',
    'value_estimation_loss',
    'l2_regularization_loss',
    'entropy_regularization_loss',
    'kl_penalty_loss'
))
def double_batch_pred2(the_model,all_inputs,specs,is_training=False):
    outer_dims = nest_utils.get_outer_array_shape(all_inputs, specs)
    all_inputs,_ = nest_utils.flatten_multi_batched_nested_tensors(all_inputs, specs)
    
    vals = the_model(all_inputs,is_training=is_training)
    vals = tf.reshape(vals,(*outer_dims,-1))
    return vals


def get_neglopacs(labels,logits):
    return losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits)

def _normalize_advantages(advantages, axes=(0,), variance_epsilon=1e-8):
  adv_mean, adv_var = tf.nn.moments(x=advantages, axes=axes, keepdims=True)
  normalized_advantages = ((advantages - adv_mean) / (tf.sqrt(adv_var) + variance_epsilon))
  return normalized_advantages

# class PPOAgent(tf_agent.TFAgent):
@gin.configurable
class PPOAgent():
  """A PPO Agent."""

  def __init__(self,
               optimizer=None,
               actor_net=None,
               value_net=None,
               observation_spec=None,
               num_actions=None,
               importance_ratio_clipping=0.2,
               lambda_value=0.95,
               discount_factor=0.97,
               entropy_regularization=0.0,
               policy_l2_reg=0.0,
               value_function_l2_reg=0.0,
               value_pred_loss_coef=0.5,
               num_epochs=10,
               use_gae=True,
               use_td_lambda_return=True,
               normalize_rewards=True,
               reward_norm_clipping=10.0,
               normalize_observations=False,
               log_prob_clipping=0.0,
               kl_cutoff_factor=2.0,
               kl_cutoff_coef=1000.0,
               initial_adaptive_kl_beta=0.0,
               adaptive_kl_target=0.01,
               adaptive_kl_tolerance=0.3,
               gradient_clipping=None,
               check_numerics=False,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):
    """Creates a PPO Agent.
    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      optimizer: Optimizer to use for the agent.
      actor_net: A function actor_net(observations, action_spec) that returns
        tensor of action distribution params for each observation. Takes nested
        observation and returns nested action.
      value_net: A function value_net(time_steps) that returns value tensor from
        neural net predictions for each observation. Takes nested observation
        and returns batch of value_preds.
      importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
        For more detail, see explanation at the top of the doc.
      lambda_value: Lambda parameter for TD-lambda computation.
      discount_factor: Discount factor for return computation.
      entropy_regularization: Coefficient for entropy regularization loss term.
      policy_l2_reg: Coefficient for l2 regularization of policy weights.
      value_function_l2_reg: Coefficient for l2 regularization of value function
        weights.
      value_pred_loss_coef: Multiplier for value prediction loss to balance with
        policy gradient loss.
      num_epochs: Number of epochs for computing policy updates.
      use_gae: If True (default False), uses generalized advantage estimation
        for computing per-timestep advantage. Else, just subtracts value
        predictions from empirical return.
      use_td_lambda_return: If True (default False), uses td_lambda_return for
        training value function. (td_lambda_return = gae_advantage +
        value_predictions)
      normalize_rewards: If true, keeps moving variance of rewards and
        normalizes incoming rewards.
      reward_norm_clipping: Value above and below to clip normalized reward.
      normalize_observations: If true, keeps moving mean and variance of
        observations and normalizes incoming observations.
      log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
        values.  Default: no clipping.
      kl_cutoff_factor: If policy KL changes more than this much for any single
        timestep, adds a squared KL penalty to loss function.
      kl_cutoff_coef: Loss coefficient for kl cutoff term.
      initial_adaptive_kl_beta: Initial value for beta coefficient of adaptive
        kl penalty.
      adaptive_kl_target: Desired kl target for policy updates. If actual kl is
        far from this target, adaptive_kl_beta will be updated.
      adaptive_kl_tolerance: A tolerance for adaptive_kl_beta. Mean KL above (1
        + tol) * adaptive_kl_target, or below (1 - tol) * adaptive_kl_target,
        will cause adaptive_kl_beta to be updated.
      gradient_clipping: Norm length to clip gradients.  Default: no clipping.
      check_numerics: If true, adds tf.debugging.check_numerics to help find
        NaN / Inf values. For debugging only.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If true, gradient summaries will be written.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    Raises:
      ValueError: If the actor_net is not a DistributionNetwork.
    """
#     if not isinstance(actor_net, network.DistributionNetwork):
#       raise ValueError(
#           'actor_net must be an instance of a DistributionNetwork.')

    tf.Module.__init__(self, name=name)
    self.pdtype =  make_pdtype(spaces.Discrete(num_actions))
    self._optimizer = optimizer
    self._actor_net = actor_net
    self._value_net = value_net
    self._observation_spec=observation_spec
    self._importance_ratio_clipping = importance_ratio_clipping
    self._lambda = lambda_value
    self._discount_factor = discount_factor
    self._entropy_regularization = entropy_regularization
    self._policy_l2_reg = policy_l2_reg
    self._value_function_l2_reg = value_function_l2_reg
    self._value_pred_loss_coef = value_pred_loss_coef

    self._num_epochs = num_epochs
    self._use_gae = use_gae
    self._use_td_lambda_return = use_td_lambda_return
    self._reward_norm_clipping = reward_norm_clipping
    self._log_prob_clipping = log_prob_clipping
    self._kl_cutoff_factor = kl_cutoff_factor
    self._kl_cutoff_coef = kl_cutoff_coef
    self._adaptive_kl_target = adaptive_kl_target
    self._adaptive_kl_tolerance = adaptive_kl_tolerance
    self._gradient_clipping = gradient_clipping or 0.0
    self._check_numerics = check_numerics

    self.train_step_counter=0
    if initial_adaptive_kl_beta > 0.0:
      # TODO(kbanoop): Rename create_variable.
      self._adaptive_kl_beta = common.create_variable('adaptive_kl_beta', initial_adaptive_kl_beta, dtype=tf.float32)
    else:
      self._adaptive_kl_beta = None

    self._reward_normalizer = None
    if normalize_rewards:
      self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(tensor_spec.TensorSpec([], tf.float32), scope='normalize_reward')

    self._observation_normalizer = None


    super(PPOAgent, self).__init__()


  @property
  def actor_net(self):
    """Returns actor_net TensorFlow template function."""
    return self._actor_net

  def _initialize(self):
    pass

  def compute_advantages(self, rewards, returns, discounts, value_preds):
    """Compute advantages, optionally using GAE.
    Based on baselines ppo1 implementation. Removes final timestep, as it needs
    to use this timestep for next-step value prediction for TD error
    computation.
    Args:
      rewards: Tensor of per-timestep rewards.
      returns: Tensor of per-timestep returns.
      discounts: Tensor of per-timestep discounts. Zero for terminal timesteps.
      value_preds: Cached value estimates from the data-collection policy.
    Returns:
      advantages: Tensor of length (len(rewards) - 1), because the final
        timestep is just used for next-step value prediction.
    """
    
    # Arg value_preds was appended with final next_step value. Make tensors
    #   next_value_preds by stripping first and last elements respectively.

    final_value_pred = value_preds[:, -1]
    value_preds = value_preds[:, :-1]

    if not self._use_gae:
      with tf.name_scope('empirical_advantage'):
        advantages = returns - value_preds
    else:
      advantages = value_ops.generalized_advantage_estimation(
        values=value_preds,
        final_value=final_value_pred,
        rewards=rewards,
        discounts=discounts,
        td_lambda=self._lambda,
        time_major=False)
    return advantages

  def get_epoch_loss(self, time_steps, actions, act_log_probs, returns,
                     normalized_advantages, action_distribution_parameters,
                     weights):
    """Compute the loss and create optimization op for one training epoch.
    All tensors should have a single batch dimension.
    Args:
      time_steps: A minibatch of TimeStep tuples.
      actions: A minibatch of actions.
      act_log_probs: A minibatch of action probabilities (probability under the
        sampling policy).
      returns: A minibatch of per-timestep returns.
      normalized_advantages: A minibatch of normalized per-timestep advantages.
      action_distribution_parameters: Parameters of data-collecting action
        distribution. Needed for KL computation.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      train_step: A train_step variable to increment for each train step.
        Typically the global_step.
      debug_summaries: True if debug summaries should be created.
    Returns:
      A tf_agent.LossInfo named tuple with the total_loss and all intermediate
        losses in the extra field contained in a PPOLossInfo named tuple.
    """
    # Evaluate the current policy on timesteps.

    # batch_size from time_steps

    # TODO(eholly): Rename policy distributions to something clear and uniform.
    #current_policy_distribution = distribution_step.action

    all_obs = time_steps.observation

    current_policy_distribution= double_batch_pred2(self._actor_net,all_obs,self._observation_spec,is_training=True)

    # Call all loss functions and add all loss values.
    value_estimation_loss = self.value_estimation_loss(time_steps, returns,weights)
    
    policy_gradient_loss = self.policy_gradient_loss(
        time_steps,
        actions,
        tf.stop_gradient(act_log_probs),
        tf.stop_gradient(normalized_advantages),
        current_policy_distribution,
        weights)


    l2_regularization_loss = tf.zeros_like(policy_gradient_loss)

    if self._entropy_regularization > 0.0:
      entropy_regularization_loss = self.entropy_regularization_loss(
        time_steps, current_policy_distribution, weights)
    else:
      entropy_regularization_loss = tf.zeros_like(policy_gradient_loss)
    
    kl_penalty_loss = self.kl_penalty_loss(
      time_steps, action_distribution_parameters, current_policy_distribution,
      weights)
    
    total_loss = (
      policy_gradient_loss + value_estimation_loss + l2_regularization_loss +
      entropy_regularization_loss + kl_penalty_loss)

    return tf_agent.LossInfo(
      total_loss,
      PPOLossInfo(
        policy_gradient_loss=policy_gradient_loss,
        value_estimation_loss=value_estimation_loss,
        l2_regularization_loss=l2_regularization_loss,
        entropy_regularization_loss=entropy_regularization_loss,
        kl_penalty_loss=kl_penalty_loss
      ))

  def compute_return_and_advantage(self, next_time_steps, value_preds):
    """Compute the Monte Carlo return and advantage.
    Normalazation will be applied to the computed returns and advantages if
    it's enabled.
    Args:
      next_time_steps: batched tensor of TimeStep tuples after action is taken.
      value_preds: Batched value prediction tensor. Should have one more entry
        in time index than time_steps, with the final value corresponding to the
        value prediction of the final state.
    Returns:
      tuple of (return, normalized_advantage), both are batched tensors.
    """
    #discounts = discounts * tf.constant(
    #  self._discount_factor, dtype=tf.float32)
  
    discounts = next_time_steps.discount * tf.constant(
        self._discount_factor, dtype=tf.float32)

    rewards = next_time_steps.reward


    # Normalize rewards if self._reward_normalizer is defined.
    if self._reward_normalizer:
      rewards = self._reward_normalizer.normalize(
        rewards, center_mean=False, clip_value=self._reward_norm_clipping)

    #print("rew_n",rewards)
    # Make discount 0.0 at end of each episode to restart cumulative sum
    #   end of each episode.
    episode_mask = common.get_episode_mask(next_time_steps)
    discounts *= episode_mask

    
    # Compute Monte Carlo returns.
    returns = value_ops.discounted_return(rewards, discounts, time_major=False)
    #print("RET",returns)
    # Compute advantages.
    advantages = self.compute_advantages(rewards, returns, discounts,
                                         value_preds)
    normalized_advantages = _normalize_advantages(advantages, axes=(0, 1))


    # Return TD-Lambda returns if both use_td_lambda_return and use_gae.
    if self._use_td_lambda_return:
      if not self._use_gae:
        logging.warning('use_td_lambda_return was True, but use_gae was '
                        'False. Using Monte Carlo return.')
      else:
        returns = tf.add(
          advantages, value_preds[:, :-1], name='td_lambda_returns')

    return returns, normalized_advantages

  def _train(self, experience,weights):
    # Get individual tensors from transitions.
    (time_steps, policy_steps_,next_time_steps) = trajectory.to_transition(experience)
    
    #observations = time_steps.observation
    actions = policy_steps_.action
    #rewards = next_time_steps.reward
    #discounts = next_time_steps.discount
    
    old_actions_distribution = policy_steps_.info

    act_log_probs = get_neglopacs(logits=old_actions_distribution, labels=actions)
    
    # Compute the value predictions for states using the current value function.


    value_preds = double_batch_pred2(self._value_net,experience.observation,self._observation_spec,is_training=True)
    value_preds = tf.squeeze(value_preds,-1)
    
    #NeedValue preds at all time_steps +1 final step obs
    #print("Weight",weights)
    #print("REW",rewards)
    #print("Dis",discounts)
    returns, normalized_advantages = self.compute_return_and_advantage(
      next_time_steps,value_preds)

    
    #print("RET",returns)
    #print(normalized_advantages)
    # Loss tensors across batches will be aggregated for summaries.
    policy_gradient_losses = []
    value_estimation_losses = []
    l2_regularization_losses = []
    entropy_regularization_losses = []
    kl_penalty_losses = []

    loss_info = None  # TODO(b/123627451): Remove.
    # For each epoch, create its own train op that depends on the previous one.
    for i_epoch in range(self._num_epochs):
      with tf.name_scope('epoch_%d' % i_epoch):


        # Build one epoch train op.
        with tf.GradientTape() as tape:
          loss_info = self.get_epoch_loss(
            time_steps, actions, act_log_probs, returns,
            normalized_advantages, old_actions_distribution, weights)#action_distribution_parameters

        variables_to_train = (
          self._actor_net.trainable_variables +
          self._value_net.trainable_variables)
        grads = tape.gradient(loss_info.loss, variables_to_train)
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = tuple(zip(grads, variables_to_train))
        if self._gradient_clipping > 0:
          grads_and_vars = eager_utils.clip_gradient_norms(
            grads_and_vars, self._gradient_clipping)




        self._optimizer.apply_gradients(
          grads_and_vars)#, global_step=self.train_step_counter)

        policy_gradient_losses.append(loss_info.extra.policy_gradient_loss)
        value_estimation_losses.append(loss_info.extra.value_estimation_loss)
        l2_regularization_losses.append(loss_info.extra.l2_regularization_loss)
        entropy_regularization_losses.append(loss_info.extra.entropy_regularization_loss)
        kl_penalty_losses.append(loss_info.extra.kl_penalty_loss)

    # After update epochs, update adaptive kl beta, then update observation
    #   normalizer and reward normalizer.
    # Compute the mean kl from previous action distribution.
    temp_ = double_batch_pred2(self._actor_net,time_steps.observation,self._observation_spec,is_training=True)
    kl_divergence = self._kl_divergence(
      time_steps, old_actions_distribution,
      temp_)
    self.update_adaptive_kl_beta(kl_divergence)

    if self._observation_normalizer:
      self._observation_normalizer.update(
        time_steps.observation, outer_dims=[0, 1])
    else:
      # TODO(b/127661780): Verify performance of reward_normalizer when obs are
      #                    not normalized
      if self._reward_normalizer:
        self._reward_normalizer.update(next_time_steps.reward,outer_dims=[0, 1])

    loss_info = tf.nest.map_structure(tf.identity, loss_info)
    return loss_info

  def l2_regularization_loss(self, debug_summaries=False):
    if self._policy_l2_reg > 0 or self._value_function_l2_reg > 0:
      with tf.name_scope('l2_regularization'):
        # Regularize policy weights.
        policy_vars_to_l2_regularize = [
          v for v in self._actor_net.trainable_weights if 'kernel' in v.name
        ]
        policy_l2_losses = [
          tf.reduce_sum(input_tensor=tf.square(v)) * self._policy_l2_reg
          for v in policy_vars_to_l2_regularize
        ]

        # Regularize value function weights.
        vf_vars_to_l2_regularize = [
          v for v in self._value_net.trainable_weights if 'kernel' in v.name
        ]
        vf_l2_losses = [
          tf.reduce_sum(input_tensor=tf.square(v)) *
          self._value_function_l2_reg for v in vf_vars_to_l2_regularize
        ]

        l2_losses = policy_l2_losses + vf_l2_losses
        total_l2_loss = tf.add_n(l2_losses, name='l2_loss')

        if self._check_numerics:
          total_l2_loss = tf.debugging.check_numerics(total_l2_loss,
                                                      'total_l2_loss')

        if debug_summaries:
          tf.compat.v2.summary.histogram(
            name='l2_loss', data=total_l2_loss, step=self.train_step_counter)
    else:
      total_l2_loss = tf.constant(0.0, dtype=tf.float32, name='zero_l2_loss')

    return total_l2_loss

  def entropy_regularization_loss(self,time_steps,current_policy_distribution,weights):
    """Create regularization loss tensor based on agent parameters."""

    if self._entropy_regularization >0.0:

      with tf.name_scope('entropy_regularization'):
        #entropy = tf.cast(
        #  common.entropy(current_policy_distribution, self.action_spec),
        #  tf.float32)
        latents = tf.math.log(current_policy_distribution)
        pd,_ = self.pdtype.pdfromlatent(latents)

        entropy_reg_loss = (
          tf.reduce_mean(input_tensor=-pd.entropy() * weights) *
          self._entropy_regularization)
       
    else:
      entropy_reg_loss = tf.constant(0.0, dtype=tf.float32, name='zero_entropy_reg_loss')


    return entropy_reg_loss

  def value_estimation_loss(self,time_steps,returns,weights):
    """Computes the value estimation loss for actor-critic training.
    All tensors should have a single batch dimension.
    Args:
      time_steps: A batch of timesteps.
      returns: Per-timestep returns for value function to predict. (Should come
        from TD-lambda computation.)
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.
    Returns:
      value_estimation_loss: A scalar value_estimation_loss loss.
    """
    observation = time_steps.observation

    value_preds = double_batch_pred2(self._value_net,observation,self._observation_spec,is_training=True)
    value_preds = tf.squeeze(value_preds,-1)

    #print("VP",value_preds)
    #print(returns)
    value_estimation_error = tf.math.squared_difference(returns, value_preds)
    value_estimation_error *= weights

    value_estimation_loss = (tf.reduce_mean(input_tensor=value_estimation_error) * self._value_pred_loss_coef)
    

    return value_estimation_loss

  def policy_gradient_loss(self,time_steps,actions,sample_action_log_probs,
                           advantages,current_policy_distribution,
                           weights):
    """Create tensor for policy gradient loss.
    All tensors should have a single batch dimension.
    Args:
      time_steps: TimeSteps with observations for each timestep.
      actions: Tensor of actions for timesteps, aligned on index.
      sample_action_log_probs: Tensor of sample probability of each action.
      advantages: Tensor of advantage estimate for each timestep, aligned on
        index. Works better when advantage estimates are normalized.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Includes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.
    Returns:
      policy_gradient_loss: A tensor that will contain policy gradient loss for
        the on-policy experience.
    """


    action_log_probs = get_neglopacs(logits =current_policy_distribution,labels = actions)
    action_log_probs= tf.cast(action_log_probs, tf.float32)
    if self._log_prob_clipping > 0.0:
      action_log_probs = tf.clip_by_value(
          action_log_probs, -self._log_prob_clipping, self._log_prob_clipping)


    # Prepare both clipped and unclipped importance ratios.
    #importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
    #ERIC: been using neglogpac instead of log_prob... so invert sample/action
    importance_ratio = tf.exp(sample_action_log_probs - action_log_probs)
    importance_ratio_clipped = tf.clip_by_value(importance_ratio,
        1 - self._importance_ratio_clipping,
        1 + self._importance_ratio_clipping)

    
    # Pessimistically choose the minimum objective value for clipped and
    #   unclipped importance ratios.
    per_timestep_objective = importance_ratio * advantages
    per_timestep_objective_clipped = importance_ratio_clipped * advantages
    per_timestep_objective_min = tf.minimum(per_timestep_objective,per_timestep_objective_clipped)

    if self._importance_ratio_clipping > 0.0:
      policy_gradient_loss = -per_timestep_objective_min
    else:
      policy_gradient_loss = -per_timestep_objective

    policy_gradient_loss = tf.reduce_mean(input_tensor=policy_gradient_loss * weights)

   
    return policy_gradient_loss

  def kl_cutoff_loss(self, kl_divergence, debug_summaries=False):
    # Squared penalization for mean KL divergence above some threshold.
    if self._kl_cutoff_factor <= 0.0:
      return tf.constant(0.0, dtype=tf.float32, name='zero_kl_cutoff_loss')
    kl_cutoff = self._kl_cutoff_factor * self._adaptive_kl_target
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    kl_over_cutoff = tf.maximum(mean_kl - kl_cutoff, 0.0)
    kl_cutoff_loss = self._kl_cutoff_coef * tf.square(kl_over_cutoff)


    return tf.identity(kl_cutoff_loss, name='kl_cutoff_loss')

  def adaptive_kl_loss(self, kl_divergence, debug_summaries=False):
    if self._adaptive_kl_beta is None:
      return tf.constant(0.0, dtype=tf.float32, name='zero_adaptive_kl_loss')

    # Define the loss computation, which depends on the update computation.
    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)
    adaptive_kl_loss = self._adaptive_kl_beta * mean_kl



    return adaptive_kl_loss

  def _kl_divergence(self, time_steps, action_distribution_parameters,current_policy_distribution):
   
    
    kl_divergence = losses.kullback_leibler_divergence(action_distribution_parameters, current_policy_distribution)

    return kl_divergence

  def kl_penalty_loss(self,time_steps,action_distribution_parameters,
                      current_policy_distribution,weights):
    """Compute a loss that penalizes policy steps with high KL.
    Based on KL divergence from old (data-collection) policy to new (updated)
    policy.
    All tensors should have a single batch dimension.
    Args:
      time_steps: TimeStep tuples with observations for each timestep. Used for
        computing new action distributions.
      action_distribution_parameters: Action distribution params of the data
        collection policy, used for reconstruction old action distributions.
      current_policy_distribution: The policy distribution, evaluated on all
        time_steps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.  Inlcudes a mask for invalid timesteps.
      debug_summaries: True if debug summaries should be created.
    Returns:
      kl_penalty_loss: The sum of a squared penalty for KL over a constant
        threshold, plus an adaptive penalty that encourages updates toward a
        target KL divergence.
    """
    kl_divergence = self._kl_divergence(time_steps,
                                        action_distribution_parameters,
                                        current_policy_distribution) * weights

 

    kl_cutoff_loss = self.kl_cutoff_loss(kl_divergence)
    adaptive_kl_loss = self.adaptive_kl_loss(kl_divergence)
    return tf.add(kl_cutoff_loss, adaptive_kl_loss, name='kl_penalty_loss')

  def update_adaptive_kl_beta(self, kl_divergence):
    """Create update op for adaptive KL penalty coefficient.
    Args:
      kl_divergence: KL divergence of old policy to new policy for all
        timesteps.
    Returns:
      update_op: An op which runs the update for the adaptive kl penalty term.
    """
    if self._adaptive_kl_beta is None:
      return tf.no_op()

    mean_kl = tf.reduce_mean(input_tensor=kl_divergence)

    # Update the adaptive kl beta after each time it is computed.
    mean_kl_below_bound = (
        mean_kl <
        self._adaptive_kl_target * (1.0 - self._adaptive_kl_tolerance))
    mean_kl_above_bound = (
        mean_kl >
        self._adaptive_kl_target * (1.0 + self._adaptive_kl_tolerance))
    adaptive_kl_update_factor = tf.case([
        (mean_kl_below_bound, lambda: tf.constant(1.0 / 1.5, dtype=tf.float32)),
        (mean_kl_above_bound, lambda: tf.constant(1.5, dtype=tf.float32)),
    ], default=lambda: tf.constant(1.0, dtype=tf.float32), exclusive=True)

    new_adaptive_kl_beta = tf.maximum(
        self._adaptive_kl_beta * adaptive_kl_update_factor, 10e-16)
    tf.compat.v1.assign(self._adaptive_kl_beta, new_adaptive_kl_beta)

    return self._adaptive_kl_beta
  
