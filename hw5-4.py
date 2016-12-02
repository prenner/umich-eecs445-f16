list_t = []

A = [[.5, .2, .3], [.2, .4, .4], [.4, .1, .5]]
phi = [[.8, .2], [.1,.9], [.5,.5]]
pi = [] #[.5, .3, .2]

list_t_ints = [] 
maxLengthList = 5
while len(list_t_ints) < maxLengthList:
	x = input("Enter an integer ( -1 to quit): ")
	if(x == -1):
		break

	list_t_ints.append(int(x))

number_of_obsv = int(input("Enter the number of observations: "))
from itertools import product
for roll in product(list_t_ints, repeat = number_of_obsv):
    list_t.append(roll)

while len(pi) < len(list_t_ints):
	x = float(input("Enter PI weights: "))
	pi.append(x)

# #Prior probability
prior = 0
likelihood = 0
sum_t = 0
max_total = 0
max_state = 0

#Note - this finds sum. Duplicate code and not efficient. Just had to get something down.
for i in list_t:
	#RESET
	prior = 0
	likelihood = 1
	index = 0

	#GO through priors
	print(i)
	val = i[0]
	val_int = int(val)
	prior = pi[val]
	for letter in i:
		j = int(letter)
		if(index >= 1):
			print(index-1)
			x = int(i[index - 1])
			print(x)
			print(j)
			prior *= A[x][j]
		index += 1

	#Reset
	index = 0

	for letter in i:
		j = int(letter)
		if(index == 0):
			likelihood *= phi[j][0]
		if(index == 1):
			likelihood *= phi[j][1]
		if(index == 2):
			likelihood *= phi[j][0]
		if(index == 3):
			likelihood *= phi[j][1]
		index += 1
	sum_t += prior*likelihood
##Sum t holds denominator

pr = 0
likelihood_T = 0
ignore_list = []
for i in list_t:
	#ignore the following:
	if i in ignore_list:
		continue
	#RESET
	likelihood = 1
	index = 0
	#Initialize prior
	prior = pi[int(i[0])]

	for letter in i:
		j = int(letter)
		if(index >= 1):
			x = int(i[index -1])
			prior *= A[x][j]
		index += 1


	index = 0

	for letter in i:
		j = int(letter)
		if(index == 0):
			likelihood *= phi[j][0]
		if(index == 1):
			likelihood *= phi[j][1]
		if(index == 2):
			likelihood *= phi[j][0]
		if(index == 3):
			likelihood *= phi[j][1]
		index += 1

	print("Combination = " + str(prior*likelihood/sum_t))

	if(max_total < prior*likelihood/sum_t):
		pr = prior
		likelihood_T = likelihood
		max_total = prior*likelihood/sum_t
		max_state = i

#State
print("State: " + str(max_state))
#Total posterior probability
print("Posterior probability: " + str(max_total))
#Likelihood
print("Likelihood: " + str(likelihood_T))
#Prior 
print("Prior: " + str(pr))

#This finds the best state. Take code and add the best state to the ignore list to calculate top 3.



