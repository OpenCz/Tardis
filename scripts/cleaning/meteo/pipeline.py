from . import loading

class Pipeline:
    def __init__(self,
                 data_vent_path : str,
                 data_parameter_path : str,
                 output_path: str = "../../../data/processed/trains/cleaned_meteo_dataset.csv",
                 report_path: str = "../../../data/processed/audit/cleaning_meteo_report.csv",
                 ):
        self.data_vent_path = data_vent_path
        self.data_parameter_path = data_parameter_path
        self.output_path = output_path
        self.report_path = report_path
        
        self.df = None
        self.original = None
    
    def run(self):

    def _load(self):
        self.df, self.original = loading.load_meteo(self.data_vent_path, self.data_parameter_path)