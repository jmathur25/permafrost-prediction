

import numpy as np


pw = 0.997 # g/m^3
pi = 0.9168 # g/cm^3
    
alt_to_def = (pw - pi) / pi


def make_simulated_data():
    rand = np.random.RandomState(7)
    
    R = 0.002 # 0.2 cm/yr
    E = 0.02 # 2 cm
    
    # Pure water column
    k_p = 2.22 # W/(m*C)
    L = 334 # J/g
    p_p = 1e6 # g/m^3
    N = k_p / (p_p * L) * 60 * 60 * 24 # m^2 / (d * C)
    
    years = [2006, 2007, 2008, 2009, 2010]
    ddt_per_year = []
    ddt_growth_per_day = (E/alt_to_def)**2 / (2 * N * 364) # last day will achieve E subsidence
    for y in years:
        ddt_per_day = np.arange(365) * ddt_growth_per_day
        ddt_bump = rand.rand() * 10.0 - 5.0 # add +/- 5 degrees of noise
        ddt_per_day[1:] += ddt_bump
        # if DDT is less than 0, it's just a freeze day, so set it to 0.
        ddt_per_day[ddt_per_day < 0] = 0.0
        ddt_per_year.append(ddt_per_day)
    
    subsidence_per_year = []
    Es = []
    for y, ddt_per_day in zip(years, ddt_per_year):
        seasonal_sub = np.sqrt(ddt_per_day * 2 * N) * alt_to_def
        delta_years = y - years[0]
        delta_days = np.arange(365) / 365
        longterm_sub = R * (delta_years + delta_days)
        subsidence_per_year.append(seasonal_sub + longterm_sub)
        Es.append(seasonal_sub[-1])
    
    # Because we randomly added noise to DDTs, the E will not be exactly what was specified.
    E_actual = np.mean(Es)
    assert abs(E_actual - E) <= 0.001 # should not deviate more than 0.1 cm from specified E
    return subsidence_per_year, ddt_per_year, years, R, E_actual


def test():
    subsidence_per_year, ddt_per_year, years, R, E = make_simulated_data()
    
    rand = np.random.RandomState(7)
    igrams = []
    N_DATA_POINTS = 20
    for _ in range(N_DATA_POINTS):
        y1, d1, y2, d2 = generate_year_day(years, rand)
        while (y2 - y1) * 365 + d2 - d1 <= 30:
            y1, d1, y2, d2 = generate_year_day(years, rand)
        igrams.append((y1, d1, y2, d2))
    
    R_pred_resalt_norm_per_year, E_pred_resalt_norm_per_year = solve_resalt(subsidence_per_year, ddt_per_year, years, igrams, True)
    R_pred_resalt_norm_across_years, E_pred_resalt_norm_across_years = solve_resalt(subsidence_per_year, ddt_per_year, years, igrams, False)

    print_stats(R_pred_resalt_norm_per_year, R, E_pred_resalt_norm_per_year, E, "resalt_norm_per_year")
    print_stats(R_pred_resalt_norm_across_years, R, E_pred_resalt_norm_across_years, E, "resalt_norm_across_years")

    print()
    

def print_stats(R_pred, R, E_Pred, E, desc):
    R_abs_err = abs(R_pred - R)
    E_abs_err = abs(E_Pred - E)
    
    R_rel_err = R_abs_err / R
    E_rel_err = E_abs_err / E
    
    R_abs_err = round(R_abs_err, 5)
    R_rel_err = round(R_rel_err, 5)
    E_abs_err = round(E_abs_err, 5)
    E_rel_err = round(E_rel_err, 5)
    print(f"For {desc}, R abs err: {R_abs_err}, R rel err: {R_rel_err}, E abs err: {E_abs_err}, E rel err: {E_rel_err}")


# TODO: merge with schaefer?
def solve_resalt(subsidence_per_year, ddt_per_year, years, igrams, norm_per_year: bool):
    if norm_per_year:
        ddt_per_year = [dpy / dpy[-1] for dpy in ddt_per_year]
    else:
        max_ddt = max([dpy[-1] for dpy in ddt_per_year])
        ddt_per_year = [dpy / max_ddt for dpy in ddt_per_year]
    lhs = []
    rhs = []
    for (y1, d1, y2, d2) in igrams:
        sub1, ddt1 = get_sub_ddt(subsidence_per_year, ddt_per_year, years, y1, d1)
        sub2, ddt2 = get_sub_ddt(subsidence_per_year, ddt_per_year, years, y2, d2)
        lhs.append(sub2 - sub1)
        delta_t = (y2 - y1) + (d2 - d1)/365
        sqrt_ddt_diff = np.sqrt(ddt2) - np.sqrt(ddt1)
        rhs.append([delta_t, sqrt_ddt_diff])
        
    rhs_pi = np.linalg.pinv(rhs)
    sol = rhs_pi @ lhs

    R_pred = sol[0]
    E_pred = sol[1]
    
    # We normalized ADDT, so it should be close to E_pred. We need to scale of E by the sqrt(DDT) per year
    # to get an estimate of the subsidence at measure time per year. Then we average those to get a sense
    # of average measured E.
    Es = []
    for dpy in ddt_per_year:
        Ei = E_pred * np.sqrt(dpy[-1])
        Es.append(Ei)
    E_pred = np.mean(Es)
    
    return R_pred, E_pred
    

def get_sub_ddt(subsidence_per_year, ddt_per_year, years, y1, d1):
    year_idx = y1 - years[0]
    day_idx = d1
    ddt = ddt_per_year[year_idx][day_idx]
    sub = subsidence_per_year[year_idx][day_idx]
    return sub, ddt

def generate_year_day(years, rand):
    y1 = years[rand.randint(0, len(years))]
    d1 = rand.randint(365)
        
    y2 = years[rand.randint(0, len(years))]
    d2 = rand.randint(365)
    return y1,d1,y2,d2
        
        
        
    
if __name__ == '__main__':
    test()
    
