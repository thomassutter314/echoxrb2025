import matplotlib
matplotlib.use('TkAgg')
from matplotlib.ticker import FormatStrFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTk, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk)
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog as fd
import threading


class App():
    def __init__(self):
        self.sim_live = False # GUI starts without the simulation initially running
        self.t1 = time.time() # record the start time
        
        # Some formatting stuff
        self.greenColor = '#50EBFF'
        self.yellowColor = '#FF00CA'
        self.blueColor = '#FCFF00'
        self.redColor = '#03F479'
        
        # Setup a tkinter gui window
        self.root = tk.Tk()
        self.root.geometry("1000x700")
        self.root.configure(bg='white')
        self.root.wm_title("RF GUI")
        # ~ self.root.iconify()
        
        framePady = 18
        
        self.controlsFrame = tk.Frame(master=self.root, width=20, height=20, padx=15, pady = framePady)
        self.controlsFrame.pack(side=tk.LEFT,anchor='nw')
        
        # ~ self.displayFrame = tk.Frame(master=self.root, width=20, height=20, padx=15, pady = framePady)
        # ~ self.displayFrame.pack(side=tk.RIGHT,anchor='nw')
        
        self.t0 = time.time()
        
        # Make buttons and entries
        self.startButton = tk.Button(
            master=self.controlsFrame,
            text="Start\nSimulation",
            width=10,
            height=2,
            fg='black', bg='white',
            command=self.runSim)
        def resetButtonFunc():
            self.sim_live = False
            self.reset = True
        self.resetButton = tk.Button(
            master=self.controlsFrame,
            text="Reset",
            width=10,
            height=2,
            fg='black', bg='white',
            command=resetButtonFunc)
        self.resetButton["state"] = "disabled"
        self.saveStateButton = tk.Button(
            master=self.controlsFrame,
            text="Save State",
            width=10,
            height=2,
            fg='black', bg='white',
            command=self.saveState)
        self.saveStateButton["state"] = "disabled"
        self.lattice_size_entry = tk.Entry(master=self.controlsFrame)
        self.particle_number_entry = tk.Entry(master=self.controlsFrame)
        self.initial_temp_entry = tk.Entry(master=self.controlsFrame)
        self.final_temp_entry = tk.Entry(master=self.controlsFrame)
        self.cooling_rate_entry = tk.Entry(master=self.controlsFrame)
        self.iterations_per_plot_entry = tk.Entry(master=self.controlsFrame)
        
        
        # Load the initial values for these entries
        settingsDict = {}
        with open('sim_settings.txt','r') as ssf:
            for line in ssf:
                settingsDict[line[:line.find(":")]] = line[line.find(":")+1:].replace('\n','')
            
        entryList = [self.lattice_size_entry,self.particle_number_entry,self.initial_temp_entry,self.final_temp_entry,self.cooling_rate_entry,self.iterations_per_plot_entry]
        entryCodes = ['lattice_size','particle_number','initial_temp','final_temp','cooling_rate','iterations_per_plot']
        entryType = [int, int, float, float, float, int]
        for key in settingsDict:
            for p in range(len(entryCodes)):
                if entryCodes[p] == key:
                    entryList[p].insert(0,entryType[p](settingsDict[key]))
                    break
        
        pady = 5
        padySep = 10
        sepHeight = 1
        sepWidth = 180
        # Pack buttons and entries
        self.startButton.pack(pady=pady)
        self.resetButton.pack(pady=pady)
        self.saveStateButton.pack(pady=pady)
        tk.Label(master=self.controlsFrame,text="Lattice Size (odd number)").pack(pady=0)
        self.lattice_size_entry.pack(pady=0)
        tk.Label(master=self.controlsFrame,text="Particle Number").pack(pady=0)
        self.particle_number_entry.pack(pady=0)
        tk.Label(master=self.controlsFrame,text="Initial Temperature").pack(pady=0)
        self.initial_temp_entry.pack(pady=0)
        tk.Label(master=self.controlsFrame,text="Final Temperature").pack(pady=0)
        self.final_temp_entry.pack(pady=0)
        tk.Label(master=self.controlsFrame,text="Cooling Rate (1e-6)").pack(pady=0)
        self.cooling_rate_entry.pack(pady=0)
        tk.Frame(master=self.controlsFrame, bd=100, relief='flat',height=sepHeight,width=sepWidth,bg='black').pack(side='top', pady=padySep)
        tk.Label(master=self.controlsFrame,text="Iteration Per Plot").pack(pady=0)
        self.iterations_per_plot_entry.pack(pady=0)
        
        
        # Exit sequence in which we rerun the simulation
        self.reset = False
        
        def on_closing():
            if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
                self.reset = False
                
                if self.sim_live:
                    self.sim_live = False
                else:
                    self.root.quit()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()
        self.root.destroy()
        
    def runSim(self):
        # Make sure the lattice size is odd
        lattice_size_temp = int(self.lattice_size_entry.get())
        if lattice_size_temp%2 ==0:
            self.lattice_size_entry.delete(0,tk.END)
            self.lattice_size_entry.insert(0,lattice_size_temp + 1)
            
        # Turn off the initial settings buttons
        self.startButton["state"] = "disabled"
        self.resetButton["state"] = "active"
        self.saveStateButton["state"] = "active"
        self.lattice_size_entry["state"] = "disabled"
        self.particle_number_entry["state"] = "disabled"
        self.initial_temp_entry["state"] = "disabled"
        self.cooling_rate_entry["state"] = "disabled"
        self.final_temp_entry["state"] = "disabled"
        
        # Save the initial settings
        settingsDict = {'lattice_size':int(self.lattice_size_entry.get()),
                    'particle_number':int(self.particle_number_entry.get()),
                    'initial_temp':float(self.initial_temp_entry.get()),
                    'final_temp':float(self.final_temp_entry.get()),
                    'cooling_rate':float(self.cooling_rate_entry.get()),
                    'iterations_per_plot':int(self.iterations_per_plot_entry.get()),}               
        with open('sim_settings.txt','w') as ssf:
            for key, value in settingsDict.items():
                ssf.write('%s:%s\n' % (key, value))
        
            
        self.pm = lattice_gas.ParticleManager(lattice_length = int(self.lattice_size_entry.get()),\
                                             particle_number = int(self.particle_number_entry.get()),\
                                             temperature = float(self.initial_temp_entry.get()),\
                                             target_temperature = float(self.final_temp_entry.get()),\
                                             cooling_rate = float(self.cooling_rate_entry.get()))
                                             
        self.dm = lattice_gas.DisplayManager(lattice_length=self.pm.lattice_length)
        self.dm.plot_intercalants(self.pm.par_indices)
        
        # Attach the matplotlib figures to a tkinter gui window
        self.canvas = FigureCanvasTkAgg(self.dm.figure, master=self.root)
        #self.toolbar = NavigationToolbar2Tk( self.canvas, self.root )
        #self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.sim_live = True
        simBatchThr = threading.Thread(target=self.simulationBatch)
        simBatchThr.start() # Process the new data asynchronously
        
    # runs the iteration_per_save number of steps before plotting the results
    def simulationBatch(self):
        iterations_per_plot = int(self.iterations_per_plot_entry.get())
        state_save_index = 0
        # ~ time.sleep(1)       
         
        while self.sim_live:
            t0 = time.time()
            for i in range(iterations_per_plot):
                self.pm.time_evolve()
            tf = time.time()
            
            state_save_index += 1
            self.dm.update_plot(self.pm.par_indices)
            self.canvas.draw()
            
            self.dm.ax.set_title('Temp = %.3f' % self.pm.temp + ', CT = %.1f' % (1e6*(tf-t0)/iterations_per_plot) + ' us/iter')

        self.root.quit()
    
    def saveState(self):
        np.savetxt('par_state.csv',self.pm.par_indices,delimiter=",")
if __name__ == '__main__':
    app = App()
    while app.reset:
        app = App()



    

    
