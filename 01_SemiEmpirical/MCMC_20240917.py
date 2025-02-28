import numpy
import os
import csv
import decimal
from decimal import Decimal
from numba import njit
import warnings 
import datetime
import time
import sys
warnings.filterwarnings("ignore")

con = decimal.getcontext()
con.prec = 100
con.Emin = -9999999999
con.Emax =  9999999999

backslash = "\\"
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
vol_wkspc = wkspc + '02_TopoData/'

# Import variables
ero_file = wkspc + 'ero_output.csv'
ero_output = numpy.genfromtxt(ero_file, delimiter=',')

w1 = numpy.where(numpy.isnan(ero_output[:,1]))[0]
w2 = numpy.where(numpy.isnan(ero_output[:,1])==False)[0]

stde_use = numpy.std(ero_output[w2,1])/100.

ero_output[w1,1] = numpy.mean(ero_output[w2,1]) + numpy.random.normal(loc=0.,scale=stde_use,size=len(w1))

date_list_file = wkspc + 'date_list.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]

date_list = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])
date_list1 = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

date_list_file = vol_wkspc + 'date_list_topo.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]

date_list_topo = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

#First survey
year_tmp = 2020
month_tmp = 10
day_tmp = 17
hour_tmp = 0
start_time = datetime.datetime(2009, 9, 15, 0) 
time_datum_start = datetime.datetime(2009, 9, 15, 0) 

#last survey
year_tmp = 2021
month_tmp = 6
day_tmp = 1
hour_tmp = 0

time_datum_last_data = date_list_topo[-1]

#first date
year_tmp = 2020
month_tmp = 1
day_tmp = 1
hour_tmp = 0

time_datum_first_data = date_list_topo[0]

print (time_datum_start)
print (time_datum_last_data)
print (time_datum_first_data)

# Define relevant time period
time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)

w_time = numpy.where((ero_output[:, 0] >= 0) & (ero_output[:, 0] <= time_tmp_last))[0]

model_out_path = wkspc + "03_Model_output/"

date_list = date_list[w_time]
date_years = numpy.array([date.year for date in date_list], dtype=numpy.int64)


# Continue with the rest of your code
events_real = numpy.array([2, 3, 2, 4, 2, 2, 1, 2, 1, 1])
events_prior = (events_real*0.0) + 0.35
num_years = 10  # From 2009 to 2018
event_counts = numpy.zeros(num_years, dtype=numpy.int64)
try:
	os.mkdir(model_out_path)
except OSError as error:
	pass

def log_lik(y,model,stdev):

	n_in = len(y)
	loglik = Decimal('0.0')
	a1 = Decimal(- ((n_in/2.)))
	a2 = Decimal(stdev**2.)
	b1 = Decimal((n_in/2.))
	b2 = Decimal(2. * numpy.pi)
	c = Decimal(numpy.sum(((y-model)**2.)/(2.*(stdev**2))))
	loglik = (a1*a2.ln()) -  (b1 * b2.ln()) - c
	-1. * float(loglik)
	return -1. * float(loglik)
	
def log_lik_vec(y,model,stdev):

	n_in = 1.
	loglik = Decimal('0.0')
	for deci in range(0,len(model)):
		a1 = Decimal(- ((n_in/2.)))
		a2 = Decimal(stdev[deci]**2.)
		b1 = Decimal((n_in/2.))
		b2 = Decimal(2. * numpy.pi)
		c = Decimal(numpy.sum(((y[deci]-model[deci])**2.)/(2.*(stdev[deci]**2))))
		d = a1 * a2.ln()
		e = b1 * b2.ln()
		f = d-e
		g = f - c
		loglik = loglik + g
		
	loglik = -1.0 * float(loglik)	
	return loglik
def log_lik_float(y,model,stdev):

	n_in = 1.
	loglik = Decimal('0.0')

	a1 = Decimal(- ((n_in/2.)))
	a2 = Decimal(stdev**2.)
	b1 = Decimal((n_in/2.))
	b2 = Decimal(2. * numpy.pi)
	c = Decimal(numpy.sum(((y-model)**2.)/(2.*(stdev**2))))
	d = a1 * a2.ln()
	e = b1 * b2.ln()
	f = d-e
	g = f - c
	loglik = loglik + g
		
	loglik = -1.0 * float(loglik)	
	return loglik


def MC(function,training_data,theta_guess,theta_priors,stepsizes,MCMC_iters):

	file_out_1 = wkspc+"03_Model_output/" + "posterior_params.csv"
	file_out_2 = wkspc+"03_Model_output/" + "loglik_output.csv"
	
	#,function_out6,function_out7,function_out8
	
	function_out,function_out2,function_out3,function_out5 = function(theta_guess)
	output_matrix_A = numpy.zeros((MCMC_iters,len(theta_guess)))
	output_matrix_A2 = numpy.zeros((MCMC_iters,len(function_out2)))
	output_matrix_A3 = numpy.zeros((MCMC_iters,len(function_out3)))
	#output_matrix_A4 = numpy.zeros((MCMC_iters,len(function_out4)))
	output_matrix_A5 = numpy.zeros((MCMC_iters,1))
	#output_matrix_A6 = numpy.zeros((MCMC_iters,len(function_out6)))

	loglik_output = numpy.zeros((MCMC_iters,1))
	accept_output = numpy.zeros((MCMC_iters,1))
	iter_vec= numpy.arange(0,MCMC_iters,1)
	index_to_change = 0	
	
	for n in range(MCMC_iters):

		if n==0:

			old_theta = theta_guess * 1.0
			new_theta  = theta_guess * 1.0
			theta_guess_orig = theta_guess * 1.0
			
			#,function_out6,function_out7,function_out8
			
			function_out,function_out2,function_out3,function_out5 = function(new_theta)
			model_err = numpy.absolute(new_theta[-1] * 1.0)
			#print(log_lik_vec(new_theta,theta_guess_orig,theta_priors)  )
			old_loglik = log_lik(function_out,training_data,model_err) #+ log_lik_vec(new_theta,theta_guess_orig,theta_priors)

		if n > 0:
			old_theta  = output_matrix_A[0,:]
			old_loglik = loglik_output[n-1,0] 
			
			new_theta[0:int(len(new_theta))] = numpy.random.normal(loc = old_theta[0:int(len(new_theta))] , scale = stepsizes[0:int(len(new_theta))] )
			#nw_theta[index_to_change] = numpy.random.normal(loc = old_theta[index_to_change], scale = stepsizes[index_to_change])
			
			while (new_theta[0] < 0.):
				new_theta[0] = numpy.random.normal(loc = old_theta[0], scale = stepsizes[0])	
			while (new_theta[2] < 0.):
				new_theta[2] = numpy.random.normal(loc = old_theta[2], scale = stepsizes[2])					
			while (new_theta[3] < 120.):
				new_theta[3] = numpy.random.normal(loc = old_theta[3], scale = stepsizes[3])
			while (new_theta[4] < 0.):
				new_theta[4] = numpy.random.normal(loc = old_theta[4], scale = stepsizes[4])					
			while ((new_theta[5] < 24.) or (new_theta[5] > 120.)):
				new_theta[5] = numpy.random.normal(loc = old_theta[5], scale = stepsizes[5])
				
			while (new_theta[-1] < 0.):
				new_theta[-1] = numpy.random.normal(loc = old_theta[-1], scale = stepsizes[-1])					

			index_to_change = index_to_change + 1
			
			if index_to_change == len(new_theta):
				index_to_change = 0	
		
		#,function_out6,function_out7,function_out8
		
		function_out,function_out2,function_out3,function_out5 = function(new_theta)
		model_err = numpy.absolute(new_theta[-1] * 1.0)

		new_loglik =log_lik(function_out,training_data,model_err) #+ log_lik_vec(new_theta,theta_guess_orig,theta_priors)
		#print(log_lik_vec(new_theta,theta_guess_orig,theta_priors)  )
		if numpy.isnan(new_loglik) == False:
			if (new_loglik < old_loglik):
				output_matrix_A[n,:]  = new_theta
				output_matrix_A2[n,:] = function_out2
				output_matrix_A3[n,:] = function_out3
				#output_matrix_A4[n,:] = function_out4
				output_matrix_A5[n,:] = function_out5
				#output_matrix_A6[n,:] = function_out6
				loglik_output[n,0] = new_loglik
				accept_output[n,0] = 1.0

			else:
				u = numpy.random.uniform(0.0,1.0)

				if (u < numpy.exp(old_loglik - new_loglik)):
					output_matrix_A[n,:]  = new_theta
					output_matrix_A2[n,:] = function_out2
					output_matrix_A3[n,:] = function_out3
					#output_matrix_A4[n,:] = function_out4
					output_matrix_A5[n,:] = function_out5
					#output_matrix_A6[n,:] = function_out6
				
					loglik_output[n,0] = new_loglik
					accept_output[n,0] = 1.0

				else:
					output_matrix_A[n,:]  = new_theta
					output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
					output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
					#output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
					output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
					
					#output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 
					loglik_output[n,0] = old_loglik
					accept_output[n,0] = 0.0

		else:
			output_matrix_A[n,:]  = new_theta
			output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
			output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
			#output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
			output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
			#output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 

			loglik_output[n,0] = old_loglik
			accept_output[n,0] = 0.0
		
		if (n>1):
			if (new_loglik < numpy.nanmin(loglik_output[0:n-1,0])):
				params_print = new_theta * 1.0
		if (n+1) % 5000 == 0:
			print("")
			print ("Iteration:",n+1)
			print ("Posterior probability:",loglik_output[n,0])
			print ("Accept rate:",numpy.mean(accept_output[0:n,0]))
			print ("Parameters:",params_print)

		if (n+1) % 50000 ==0:

			with open(file_out_1, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(output_matrix_A)					
			with open(file_out_2, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(loglik_output)	
	'''		
	with open(file_out_1, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(output_matrix_A)					
	with open(file_out_2, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(loglik_output)	
	'''
	
	#,output_matrix_A6
	
	return output_matrix_A,output_matrix_A2,output_matrix_A3,output_matrix_A5,loglik_output


def MC_fit(function,training_data,theta_guess,theta_priors,stepsizes,MCMC_iters):
	
	global date_years
	
	total_time = 0.0
	file_out_1 = wkspc+"03_Model_output/" + "posterior_params.csv"
	file_out_2 = wkspc+"03_Model_output/" + "loglik_output.csv"
	
	#,function_out6,function_out7,function_out8
	
	function_out,function_out2,function_out3,function_out5 = function(theta_guess)
	output_matrix_A = numpy.zeros((MCMC_iters,len(theta_guess)))
	#output_matrix_A2 = numpy.zeros((MCMC_iters,len(function_out2)))
	#output_matrix_A3 = numpy.zeros((MCMC_iters,len(function_out3)))
	#output_matrix_A4 = numpy.zeros((MCMC_iters,len(function_out4)))
	#output_matrix_A5 = numpy.zeros((MCMC_iters,1))
	#output_matrix_A6 = numpy.zeros((MCMC_iters,len(function_out6)))

	loglik_output = numpy.zeros((MCMC_iters,1))
	accept_output = numpy.zeros((MCMC_iters,1))
	iter_vec= numpy.arange(0,MCMC_iters,1)
	index_to_change = 0	
	a1 = 100000000
	for n in range(MCMC_iters):
		iteration_start_time = time.time()
		if n==0:

			old_theta = theta_guess * 1.0
			new_theta  = theta_guess * 1.0
			theta_guess_orig = theta_guess * 1.0
			
			#,function_out6,function_out7,function_out8
			
			function_out,function_out2,function_out3,function_out5 = function(new_theta)
						
			@njit
			def compute_event_counts(function_out2, w_time, ero_output, date_years, new_theta,event_counts):

				low_val = function_out2[-1]

				for n2 in range(len(function_out2)):
					idx = len(function_out2) - (n2 + 1)
					valu = function_out2[idx]

					if valu < low_val:
						low_val = valu

						# Ensure index ranges are within bounds
						ero_idx_start = max(idx - int(new_theta[11]), 0)
						ero_idx_end = min(idx + int(new_theta[12]), len(w_time))
						semi_idx_start = max(idx - int(new_theta[11]), 0)
						semi_idx_end = min(idx + int(new_theta[11]), len(function_out2))

						ero_min = numpy.min(ero_output[w_time[ero_idx_start:ero_idx_end], 1])
						semi_min = numpy.min(function_out2[semi_idx_start:semi_idx_end])

						if (ero_min < new_theta[10]) and (valu <= semi_min):
							event_year = date_years[idx]
							year_index = event_year - 2009
							if 0 <= year_index < num_years:
								event_counts[year_index] += 1

				return event_counts

			# Now call the function
			event_counts_vector = numpy.array(compute_event_counts(function_out2, w_time, ero_output, date_years, new_theta,event_counts*0))

			model_err = numpy.abs(new_theta[-1])

			old_loglik = log_lik(function_out, training_data, model_err) + log_lik_vec(event_counts_vector, events_real, events_prior)


		if n > 0:
			old_theta  = output_matrix_A[n-1,:]
			old_loglik = loglik_output[n-1,0] 
			
			#new_theta[0:int(len(new_theta))] = numpy.random.normal(loc = old_theta[0:int(len(new_theta))] , scale = stepsizes[0:int(len(new_theta))] )
			new_theta[index_to_change] = numpy.random.normal(loc = old_theta[index_to_change], scale = stepsizes[index_to_change])
			while (new_theta[0] < 0.):
				new_theta[0] = numpy.random.normal(loc = old_theta[0], scale = stepsizes[0])	
			while (new_theta[2] < 0.):
				new_theta[2] = numpy.random.normal(loc = old_theta[2], scale = stepsizes[2])					
			while (new_theta[3] < 500.):
				new_theta[3] = numpy.random.normal(loc = old_theta[3], scale = stepsizes[3])
			while (new_theta[4] < 0.):
				new_theta[4] = numpy.random.normal(loc = old_theta[4], scale = stepsizes[4])					
			while ((new_theta[5] < 100.) or (new_theta[5] > 500.)):
				new_theta[5] = numpy.random.normal(loc = old_theta[5], scale = stepsizes[5])
			
			
			while (new_theta[9] > -10.):
				new_theta[9] = numpy.random.normal(loc = old_theta[9], scale = stepsizes[9])	
			while (new_theta[10] > -5.):
				new_theta[10] = numpy.random.normal(loc = old_theta[10], scale = stepsizes[10])	
			while (new_theta[11] < 0.):
				new_theta[11] = numpy.random.normal(loc = old_theta[11], scale = stepsizes[11])	
			while (new_theta[12] < 0.):
				new_theta[12] = numpy.random.normal(loc = old_theta[12], scale = stepsizes[12])					
			index_to_change = index_to_change + 1
			
			if index_to_change == len(new_theta):
				index_to_change = 0	
		
		#,function_out6,function_out7,function_out8
		
		function_out,function_out2,function_out3,function_out5 = function(new_theta)
		model_err = numpy.absolute(new_theta[-1] * 1.0)

		@njit
		def compute_event_counts(function_out2, w_time, ero_output, date_years, new_theta,event_counts):
			
			low_val = function_out2[-1]

			for n2 in range(len(function_out2)):
				idx = len(function_out2) - (n2 + 1)
				valu = function_out2[idx]

				if valu < low_val:
					low_val = valu

					# Ensure index ranges are within bounds
					ero_idx_start = max(idx - int(new_theta[11]), 0)
					ero_idx_end = min(idx + int(new_theta[12]), len(w_time))
					semi_idx_start = max(idx - int(new_theta[11]), 0)
					semi_idx_end = min(idx + int(new_theta[11]), len(function_out2))

					ero_min = numpy.min(ero_output[w_time[ero_idx_start:ero_idx_end], 1])
					semi_min = numpy.min(function_out2[semi_idx_start:semi_idx_end])

					if (ero_min < new_theta[10]) and (valu <= semi_min):
						event_year = date_years[idx]
						year_index = event_year - 2009
						if 0 <= year_index < num_years:
							event_counts[year_index] += 1

			return event_counts

		# Now call the function
		event_counts_vector = numpy.array(compute_event_counts(function_out2, w_time, ero_output, date_years, new_theta,event_counts*0))

		model_err = numpy.abs(new_theta[-1])
		
		
		new_loglik =log_lik(function_out,training_data,model_err) + log_lik_vec(event_counts_vector,events_real,events_prior)
		
		a1_tmp = log_lik_vec(event_counts_vector,events_real,events_prior)
		
		#print(log_lik_vec(new_theta,theta_guess_orig,theta_priors)  )
		if numpy.isnan(new_loglik) == False:
			if (new_loglik < old_loglik):
				output_matrix_A[n,:]  = new_theta
				#output_matrix_A2[n,:] = function_out2
				#output_matrix_A3[n,:] = function_out3
				#output_matrix_A4[n,:] = function_out4
				#output_matrix_A5[n,:] = function_out5
				#output_matrix_A6[n,:] = function_out6
				loglik_output[n,0] = new_loglik
				accept_output[n,0] = 1.0
				a1 = a1_tmp
			else:
				u = numpy.random.uniform(0.0,1.0)

				if (u < numpy.exp(old_loglik - new_loglik)):
					output_matrix_A[n,:]  = new_theta
					#output_matrix_A2[n,:] = function_out2
					#output_matrix_A3[n,:] = function_out3
					#output_matrix_A4[n,:] = function_out4
					#output_matrix_A5[n,:] = function_out5
					#output_matrix_A6[n,:] = function_out6
				
					loglik_output[n,0] = new_loglik
					accept_output[n,0] = 1.0
					a1 = a1_tmp
				else:
					output_matrix_A[n,:]  = old_theta
					#output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
					#output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
					#output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
					#output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
					
					#output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 
					loglik_output[n,0] = old_loglik
					accept_output[n,0] = 0.0

		else:
			output_matrix_A[n,:]  = old_theta
			#output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
			#output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
			#output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
			#output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
			#output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 

			loglik_output[n,0] = old_loglik
			accept_output[n,0] = 0.0
		
		if (n>1):
			if (new_loglik < numpy.nanmin(loglik_output[0:n-1,0])):
				params_print = new_theta * 1.0
				a1 = a1_tmp
		if (n+1) % 5000 == 0:
			print("")
			print ("Iteration:",n+1)
			print ("Posterior probability:",loglik_output[n,0])
			print ("Accept rate:",numpy.mean(accept_output[0:n,0]))
			print ("Parameters:",params_print)

		if (n+1) % 50000 ==0:

			with open(file_out_1, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(output_matrix_A)					
			with open(file_out_2, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(loglik_output)	

		# Compute iteration time and average time per iteration
		iteration_time = time.time() - iteration_start_time  # Compute time taken for the iteration
		total_time += iteration_time  # Add to total_time
		average_time = total_time / (n + 1)  # Compute average time per iteration
		
		if n >2:
			sys.stdout.write(f"\rAverage time per iteration after {n + 1} iterations: {round(average_time,2)} seconds, loglik: {round(loglik_output[n,0],2)}, rate: {round(numpy.mean(accept_output[0:n,0]),2)} , storm rate: {round(a1,2)}   ")


	'''		
	with open(file_out_1, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(output_matrix_A)					
	with open(file_out_2, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(loglik_output)	
	'''

	#,output_matrix_A6
	
	return output_matrix_A,loglik_output
