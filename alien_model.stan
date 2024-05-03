
data {
    int<lower=1> ntrials;  // number of trials
    int<lower=1> nfeatures;  // number of predefined relevant features
    array[ntrials] int<lower=0, upper=1> cat_safe; // true responses on a trial by trial basis
    array[ntrials] int<lower=0, upper=1> y;  // decisions on a trial by trial basis
    array[ntrials, nfeatures] real obs; // stimuli as vectors of features
    real<lower=0, upper=1> b;  // initial bias for danger over safety

    // priors
    vector[nfeatures] weight_prior_values;  // concentration parameters for dirichlet distribution <lower=1>
    array[2] real c_prior_values;  // mean and variance for logit-normal distribution
}

transformed data {
    array[ntrials] int<lower=0, upper=1> cat_dan; // dummy variable for category dangerous over cat safe
    array[sum(cat_safe)] int<lower=1, upper=ntrials> cat_safe_idx; // array of which stimuli are cat safe
    array[ntrials-sum(cat_safe)] int<lower=1, upper=ntrials> cat_dan_idx; //  array of which stimuli are cat dan
    int idx_safe = 1; // Initializing 
    int idx_dan = 1;
    for (i in 1:ntrials){
        cat_dan[i] = abs(cat_safe[i]-1);

        if (cat_safe[i]==1){
            cat_safe_idx[idx_safe] = i;
            idx_safe +=1;
        } else {
            cat_dan_idx[idx_dan] = i;
            idx_dan += 1;
        }
    }
}

parameters {
    simplex[nfeatures] weight;  // simplex means sum(w)=1
    real logit_c;
}

transformed parameters {
    // parameter c 
    real<lower=0, upper=2> c = inv_logit(logit_c)*2;  // times 2 as c is bounded between 0 and 2

    // parameter r (probability of response = category safe)
    array[ntrials] real<lower=0.0001, upper=0.9999> r;
    array[ntrials] real rr;

    for (i in 1:ntrials) {

        // calculate distance from obs to all exemplars
        array[(i-1)] real exemplar_sim;
        for (e in 1:(i-1)){
            array[nfeatures] real tmp_dist;
            for (j in 1:nfeatures) {
                tmp_dist[j] = weight[j]*abs(obs[e,j] - obs[i,j]);
            }
            exemplar_sim[e] = exp(-c * sum(tmp_dist));
        }

        if (sum(cat_safe[:(i-1)])==0 || sum(cat_dan[:(i-1)])==0){  // if there are no examplars in one of the categories
            r[i] = 0.5;

        } else {
            // calculate similarity
            array[2] real similarities;
            
            array[sum(cat_safe[:(i-1)])] int tmp_idx_safe = cat_safe_idx[:sum(cat_safe[:(i-1)])];
            array[sum(cat_dan[:(i-1)])] int tmp_idx_dan = cat_dan_idx[:sum(cat_dan[:(i-1)])];
            similarities[1] = sum(exemplar_sim[tmp_idx_safe]);
            similarities[2] = sum(exemplar_sim[tmp_idx_dan]);

            // calculate r[i]
            rr[i] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr[i] > 0.9999){
                r[i] = 0.9999;
            } else if (rr[i] < 0.0001) {
                r[i] = 0.0001;
            } else if (rr[i] > 0.0001 && rr[i] < 0.9999) {
                r[i] = rr[i];
            } else {
                r[i] = 0.5;
            }
        }
    }
}

model {
    // Priors
    target += dirichlet_lpdf(weight | weight_prior_values);
    target += normal_lpdf(logit_c | c_prior_values[1], c_prior_values[2]);
    
    
    // Decision Data
    target += bernoulli_lpmf(y | r);
}

generated quantities {
    // priors
    simplex[nfeatures] weight_prior = dirichlet_rng(weight_prior_values);
    real logit_c_prior = normal_rng(c_prior_values[1], c_prior_values[2]);
    real<lower=0, upper=2> c_prior = inv_logit(logit_c_prior)*2;

    // prior pred
    array[ntrials] real<lower=0, upper=1> r_prior;
    array[ntrials] real rr_prior;
    for (i in 1:ntrials) {

        // calculate distance from obs to all exemplars
        array[(i-1)] real exemplar_dist;
        for (e in 1:(i-1)){
            array[nfeatures] real tmp_dist;
            for (j in 1:nfeatures) {
                tmp_dist[j] = weight_prior[j]*abs(obs[e,j] - obs[i,j]);
            }
            exemplar_dist[e] = sum(tmp_dist);
        }

        if (sum(cat_safe[:(i-1)])==0 || sum(cat_dan[:(i-1)])==0){  // if there are no examplars in one of the categories
            r_prior[i] = 0.5;

        } else {
            // calculate similarity
            array[2] real similarities;
            
            array[sum(cat_safe[:(i-1)])] int tmp_idx_safe = cat_safe_idx[:sum(cat_safe[:(i-1)])];
            array[sum(cat_dan[:(i-1)])] int tmp_idx_dan = cat_dan_idx[:sum(cat_dan[:(i-1)])];
            similarities[1] = exp(-c_prior * sum(exemplar_dist[tmp_idx_safe]));
            similarities[2] = exp(-c_prior * sum(exemplar_dist[tmp_idx_dan]));

            // calculate r[i]
            rr_prior[i] = (b*similarities[1]) / (b*similarities[1] + (1-b)*similarities[2]);

            // to make the sampling work
            if (rr_prior[i] == 1){
                r_prior[i] = 0.9999;
            } else if (rr_prior[i] == 0) {
                r_prior[i] = 0.0001;
            } else if (rr_prior[i] > 0 && rr_prior[i] < 1) {
                r_prior[i] = rr_prior[i];
            } else {
                r_prior[i] = 0.5;
            }
        }
    }

    array[ntrials] int<lower=0, upper=1> priorpred = bernoulli_rng(r_prior);

    // posterior pred
    array[ntrials] int<lower=0, upper=1> posteriorpred = bernoulli_rng(r);
    array[ntrials] int<lower=0, upper=1> posteriorcorrect;
    for (i in 1:ntrials) {
        if (posteriorpred[i] == cat_safe[i]) {
            posteriorcorrect[i] = 1;
        } else {
            posteriorcorrect[i] = 0;
        }
    }
    
    
    // log likelihood
   array[ntrials] real log_lik;

   for (i in 1:ntrials) {
        log_lik[i] = bernoulli_lpmf(y[i] | r[i]);
   }

}
