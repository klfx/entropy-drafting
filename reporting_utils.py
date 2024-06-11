class DecodingMetricPoint:
    def __init__(
            self,
            token_id: int,
            is_prompt: bool = False,
            token_id_draft: int = None,
            cross_entropy: float = None,
            entropy_target: float = None,
            entropy_draft: float = None
    ):
        """
        This object can be used to store metrics for a single decoded token.

        Args:
            token_id (int): Token Id from the Target Distribution is Primariy Identifier
            is_prompt (bool): Whether the token is a prompt token.
            only_target (bool): Whether the token is only in the target distribution.
            token_id_draft (int): The token id of the draft token. (To check if it matches)
            cross_entropy (float): The cross entropy of target and draft distributions.
            entropy_target (float): The entropy of the target distribution.
            entropy_draft (float): The entropy of the draft distribution.
        
        """
        self.token_id = token_id
        self.is_prompt = is_prompt
        self.token_id_draft = token_id_draft
        self.cross_entropy = cross_entropy
        self.entropy_target = entropy_target
        self.entropy_draft = entropy_draft

    def __str__(self):
        return f"Token ID: {self.token_id}, Is Prompt: {self.is_prompt}, Token ID Draft: {self.token_id_draft}, Cross Entropy: {self.cross_entropy}, Entropy Target: {self.entropy_target}, Entropy Draft: {self.entropy_draft}"


class DecodingMetricsLogger:
    def __init__ (self, run_name='Unnamed Run'):
        """
        This object can be used to store and retrieve a sequence of DecodingMetricPoints
        """
        self.run_name = run_name
        self.point_list = []

    def log(self, point: DecodingMetricPoint):
        self.point_list.append(point)

    def get(self, metric:str = None):
        if metric:
            raise NotImplementedError("Metric Selection tbd")
        
        # Unpack DecodingMetricPoint

        token_id = [point.token_id for point in self.point_list]
        is_prompt = [point.is_prompt for point in self.point_list]
        token_id_draft = [point.token_id_draft for point in self.point_list]
        cross_entropy = [point.cross_entropy for point in self.point_list]
        entropy_target = [point.entropy_target for point in self.point_list]
        entropy_draft = [point.entropy_draft for point in self.point_list]

        return {
            'token_id': token_id,
            'is_prompt': is_prompt,
            'token_id_draft': token_id_draft,
            'cross_entropy': cross_entropy,
            'entropy_target': entropy_target,
            'entropy_draft': entropy_draft
        }

    def len(self):
        return len(self.point_list)
    
    def prompt_present(self):
        return any([point.is_prompt for point in self.point_list])


class TimeMetricsLogger:
    """
    This object can be used to log time required for forward pass on draft and target model. 
    """
    def __init__(self, run_name:str = 'Unnamed Run', device_name:str = 'unknown device'):
        self.run_name = run_name
        self.device_name = device_name
        self.log_list = []

    def log(self, draft_time: float, draft_len: int, target_time: float):
        iteration = len(self.log_list) + 1

        self.log_list.append({
            'iteration': iteration,
            'draft_time': draft_time,
            'draft_len': draft_len,
            'target_time': target_time
        })
    
    def get(self):
        """
        Returns list with draft mean per token (draft_time / draft_len)
        """
        for i in self.log_list:
            if i['draft_len'] == 0:
                i['draft_mean'] = 0
            else:
                i['draft_mean'] = i['draft_time'] / i['draft_len']

        return self.log_list
    
    def get_device_name(self):
        return self.device_name