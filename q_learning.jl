using POMDPPolicies: FunctionPolicy
using CommonRLInterface: actions, act!, observe, reset!, AbstractEnv, observations, terminated
using StaticArrays: SA
using DMUStudent.HW4: GridWorld, render 

env = GridWorld()

mutable struct QLearning
	Q 
	γ 
	ϵ 
	α 
	λ
	act 
end

function (QL::QLearning)(env)
#function Q_Learning!(QL,env)
	action_counts = 0	# Number of times we took an action
	rew_check = 0
	# use epsilon greedy to select an action
	function epsilon_greedy(s)
		if rand() < QL.ϵ
			return rand(QL.act)
		else
			return QL.act[argmax([QL.Q[(s, a)] for a in QL.act])]
		end
	end		

	
	function softmax(s) 

		Dr = 0 
		for ac in QL.act
			Dr += exp(QL.λ*QL.Q[(s, ac)])
		end
		#@show Dr
		if Dr == 4.0
			return rand(QL.act)
		else
			thetha = []
			for ac in QL.act
				thetha_estimate = round(10*(exp(QL.λ*QL.Q[(s,ac)])/Dr))
				#@show thetha_estimate
				for rep in 1:thetha_estimate
						push!(thetha, ac) 
				end
				
			end
			#@show length(thetha)
			return rand(thetha)
		end
	end

	# Start with some state
	s = observe(env) 

	# use epsilon greedy to select an action
	a = epsilon_greedy(s) 
	#a = softmax(s)
	# A new state and reward will be generated based on action taken
	r = act!(env, a) 
	action_counts+=1
	sp = observe(env)
	
	
	# Keep repeating the process of generating new state and reward till terminal state is reached
	# and update Q value averages
	
	while !terminated(env)

		best_act = QL.act[argmax([QL.Q[(sp, a)] for a in QL.act])]
		
		QL.Q[(s,a)] += QL.α*(r + QL.γ*QL.Q[(sp,best_act)] - QL.Q[(s, a)]) #Incremental estimate of Q
		
		s = sp
		a = epsilon_greedy(s)
		#a = softmax(s)
		r = act!(env, a)
		action_counts+=1
		sp = observe(env)
		end
	
	QL.Q[(s,a)] += QL.α*(r - QL.Q[(s, a)])	# Collect reward of reaching terminal state
	
	return action_counts
	  
end


# Initialize Q learning object
QL = QLearning(Dict(), 1, 0.1, 0.01, 10, actions(env)) 

# Initialize Q tables with 0
(l,b) = env.size
for i in 1:l
	for j in 1:b
		for a in QL.act
			QL.Q[(SA[i,j],a)] = 0.0
		end
	end
end


# Evaluating learned policy - Evaluation episode
function  evaluate(env, QL)
	rsum = 0
	for ii = 1:1000
		reset!(env)
		while !terminated(env)
			s = observe(env)
			eval_act = QL.act[argmax([QL.Q[(s, a)] for a in QL.act])]
			rsum += act!(env, eval_act)
		end
	end
	return rsum/1000
	
end

# Running Training episodes
n_episodes = 5*10000
average_reward = []
act_call_track = [] # keep track of calls to actions for every learning episode
time_track = [] # keep track of time to learn each eval policy
start = time() # Start of Training

for i in 1:n_episodes
	reset!(env)
	QL.ϵ = 1-i/n_episodes
	global actions_count = QL(env)
	@show i
	if (i%100==0)
		@show QL.ϵ
		push!(act_call_track,actions_count)
		push!(time_track,time()-start)
		push!(average_reward,evaluate(env,QL))
	end
end

render(env; color=s->maximum(QL.Q[(s, a)] for a in QL.act))
