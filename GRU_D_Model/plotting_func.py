import matplotlib.pyplot as plt


def plot_vitals_labels (episode_id, df):
    data = df[df['episode_id'] == episode_id]
    fig, ax1 = plt.subplots()
    ti = data['anchor_time']
    # Plot the first dataset with ax1

    line1, = ax1.plot(ti,  data['sbp'], linewidth = 0.5, label='SBP (mmHg)')
    line2, = ax1.plot(ti,  data['dbp'], linewidth = 0.5, label='DBP(mmHg)')
    line3, = ax1.plot(ti,  data['hr'], linewidth = 0.5, label='HR (bpm)')
    line4, = ax1.plot(ti,  data['temp'], linewidth = 0.5, label='Temp ($^o$C)')
    line5, = ax1.plot(ti,  data['rr'], linewidth = 0.5, label='RR (min$^-$$^1$)')
    line6, = ax1.plot(ti,  data['spo2'], linewidth = 0.5, label='SpO$_2$ (%)')
    line7, = ax1.plot(ti,  data['flow'], linewidth = 0.5, label='O$_2$ Flow Rate (L/min)')

    ax1.set_xlabel('Time Since Admission (Min)')
    ax1.set_ylabel('Vitals')
    ax1.legend(handles=[line1, line2, line3, line4, line5, line6, line7], loc='upper left', bbox_to_anchor=(1.1, 0.8))
    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    a = data[data['outcome_in_24hrs'] == 0]
    d = data[data['outcome_in_24hrs'] == 1]

    ax2.scatter(a['anchor_time'], a['outcome_in_24hrs'], color = 'forestgreen', marker= 'X', label='Y2 Data')
    ax2.scatter(d['anchor_time'], d['outcome_in_24hrs'], color = 'firebrick', marker= 'X', label='Y2 Data')
    ax2.set_ylabel('Ground Truth Labels')
    ax2.set_yticks([0,1], labels= ['Stable', 'Unstable'])
    ax2.set_ylim(-0.05,1.05)
    
    ax2.set_title('Vital Signs with Labels of Episode ID: ' + str(episode_id))
    plt.show()


def plot_vitals_gru_d_prob (df, episodes, test_idx, prob_seq, lengths, idx):

    episode_id = int(episodes[test_idx[idx]])
    data = df[df['episode_id'] == episode_id]

    fig, ax1 = plt.subplots()
    ti = data['anchor_time']


    prob_seq = gru_d_prob_seq_extractor (prob_seq, lengths, idx)
  
    # Plot the Vital Signs with ax1

    line1, = ax1.plot(ti,  data['sbp'], linewidth = 0.5, label='SBP (mmHg)')
    line2, = ax1.plot(ti,  data['dbp'], linewidth = 0.5, label='DBP(mmHg)')
    line3, = ax1.plot(ti,  data['hr'], linewidth = 0.5, label='HR (bpm)')
    line4, = ax1.plot(ti,  data['temp'], linewidth = 0.5, label='Temp ($^o$C)')
    line5, = ax1.plot(ti,  data['rr'], linewidth = 0.5, label='RR (min$^-$$^1$)')
    line6, = ax1.plot(ti,  data['spo2'], linewidth = 0.5, label='SpO$_2$ (%)')
    line7, = ax1.plot(ti,  data['flow'], linewidth = 0.5, label='O$_2$ Flow Rate (L/min)')

    ax1.set_xlabel('Time Since Admission (Min)')
    ax1.set_ylabel('Vitals')

    # Create a second y-axis for GRU-D Model Output
    ax2 = ax1.twinx()

    line8, = ax2.plot(ti, prob_seq,  color = 'k', linestyle = '-.', label='Model Output')
    line9, = ax2.plot([ti.min(), ti.max()], [0.5,0.5], linestyle = '--', color = 'firebrick', label='Model Threshold')
    ax2.set_ylabel('Probability of Adverse Outcome')

    ax2.set_ylim(-0.05,1.05)
    ax2.set_title('GRU-D Output with Episode ID: '+ str(episode_id))
    ax1.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], loc='upper left', bbox_to_anchor=(1.1, 0.8))
    plt.show()


def gru_d_prob_seq_extractor (prob_seq, lengths, idx):
    assert idx < len(lengths), 'Must Valid Index within Test Set'
    assert idx >= 0, 'Must be Valid Index'
   
    ending_idx = lengths.cumsum()
    if idx == 0:
        return prob_seq[0:ending_idx[idx]]
    else:
        start = ending_idx[(idx-1)]
        return prob_seq[start: ending_idx[idx]]

def plot_vitals_news (df, episodes, test_idx, idx):

    episode_id = int(episodes[test_idx[idx]])
    data = df[df['episode_id'] == episode_id]

    fig, ax1 = plt.subplots()
    ti = data['anchor_time']

    
    news_seq = data['news2']
  
    # Plot the Vital Signs with ax1

    line1, = ax1.plot(ti,  data['sbp'], linewidth = 0.5, label='SBP (mmHg)')
    line2, = ax1.plot(ti,  data['dbp'], linewidth = 0.5, label='DBP(mmHg)')
    line3, = ax1.plot(ti,  data['hr'], linewidth = 0.5, label='HR (bpm)')
    line4, = ax1.plot(ti,  data['temp'], linewidth = 0.5, label='Temp ($^o$C)')
    line5, = ax1.plot(ti,  data['rr'], linewidth = 0.5, label='RR (min$^-$$^1$)')
    line6, = ax1.plot(ti,  data['spo2'], linewidth = 0.5, label='SpO$_2$ (%)')
    line7, = ax1.plot(ti,  data['flow'], linewidth = 0.5, label='O$_2$ Flow Rate (L/min)')

    ax1.set_xlabel('Time Since Admission (Min)')
    ax1.set_ylabel('Vitals')

    # Create a second y-axis for EWS Score
    ax2 = ax1.twinx()

    line8, = ax2.plot(ti, news_seq,  color = 'k', linestyle = '-.', label='NEWS2 Score')
    line9, = ax2.plot([ti.min(), ti.max()], [7,7], linestyle = '--', color = 'firebrick', label = 'Threshold')
    ax2.set_ylabel('NEWS2 Score')

    ax2.set_ylim(-2,21)
    ax2.set_title('NEWS2 Output on Episode ID: '+ str(episode_id))
    ax1.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], loc='upper left', bbox_to_anchor=(1.1, 0.8))
    plt.show()

def plot_vitals_mews (df, episodes, test_idx, idx):

    episode_id = int(episodes[test_idx[idx]])
    data = df[df['episode_id'] == episode_id]
    fig, ax1 = plt.subplots()
    ti = data['anchor_time']
    news_seq = data['mews']
  
    # Plot the Vital Signs with ax1

    line1, = ax1.plot(ti,  data['sbp'], linewidth = 0.5, label='SBP (mmHg)')
    line2, = ax1.plot(ti,  data['dbp'], linewidth = 0.5, label='DBP(mmHg)')
    line3, = ax1.plot(ti,  data['hr'], linewidth = 0.5, label='HR (bpm)')
    line4, = ax1.plot(ti,  data['temp'], linewidth = 0.5, label='Temp ($^o$C)')
    line5, = ax1.plot(ti,  data['rr'], linewidth = 0.5, label='RR (min$^-$$^1$)')
    line6, = ax1.plot(ti,  data['spo2'], linewidth = 0.5, label='SpO$_2$ (%)')
    line7, = ax1.plot(ti,  data['flow'], linewidth = 0.5, label='O$_2$ Flow Rate (L/min)')

    ax1.set_xlabel('Time Since Admission (Min)')
    ax1.set_ylabel('Vitals')

    # Create a second y-axis for EWS Score
    ax2 = ax1.twinx()

    line8, = ax2.plot(ti, news_seq,  color = 'k', linestyle = '-.', label='MEWS Score')
    line9, = ax2.plot([ti.min(), ti.max()], [5,5], linestyle = '--', color = 'firebrick', label = 'Threshold')
    ax2.set_ylabel('MEWS Score')

    ax2.set_ylim(-1,15)
    ax2.set_title('MEWS Output on Episode ID: '+ str(episode_id))
    ax1.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], loc='upper left', bbox_to_anchor=(1.1, 0.8))
    plt.show()

def plot_vitals_ecart (df, episodes, test_idx, idx):

    episode_id = int(episodes[test_idx[idx]])
    data = df[df['episode_id'] == episode_id]

    fig, ax1 = plt.subplots()
    ti = data['anchor_time']
 
    news_seq = data['ecart']
  
    # Plot the Vital Signs with ax1

    line1, = ax1.plot(ti,  data['sbp'], linewidth = 0.5, label='SBP (mmHg)')
    line2, = ax1.plot(ti,  data['dbp'], linewidth = 0.5, label='DBP(mmHg)')
    line3, = ax1.plot(ti,  data['hr'], linewidth = 0.5, label='HR (bpm)')
    line4, = ax1.plot(ti,  data['temp'], linewidth = 0.5, label='Temp ($^o$C)')
    line5, = ax1.plot(ti,  data['rr'], linewidth = 0.5, label='RR (min$^-$$^1$)')
    line6, = ax1.plot(ti,  data['spo2'], linewidth = 0.5, label='SpO$_2$ (%)')
    line7, = ax1.plot(ti,  data['flow'], linewidth = 0.5, label='O$_2$ Flow Rate (L/min)')

    ax1.set_xlabel('Time Since Admission (Min)')
    ax1.set_ylabel('Vitals')

    # Create a second y-axis for EWS Score
    ax2 = ax1.twinx()

    line8, = ax2.plot(ti, news_seq,  color = 'k', linestyle = '-.', label='eCART Score')
    line9, = ax2.plot([ti.min(), ti.max()], [20,20], linestyle = '--', color = 'firebrick', label = 'Threshold')
    ax2.set_ylabel('eCART Score')
    ax2.set_ylim(-3,50)
    ax2.set_title('eCART Output on Episode ID: '+ str(episode_id))
    ax1.legend(handles=[line1, line2, line3, line4, line5, line6, line7, line8, line9], loc='upper left', bbox_to_anchor=(1.1, 0.8))
    plt.show()