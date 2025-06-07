class GNN(torch.nn.Module):
    def __init__(self, user_dim, segment_dim, nudge_dim, hidden_dim):
        super(GNN, self).__init__()
        
        # Message passing layers with Xavier initialization
        self.message_passing1 = MessagePassing(user_dim + segment_dim + nudge_dim, hidden_dim)
        self.message_passing2 = MessagePassing(hidden_dim, hidden_dim)
        
        # Initialize weights using Xavier initialization
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.message_passing1, self.message_passing2]:
            if hasattr(layer, 'lin'):  # Check if layer has linear transformation
                nn.init.xavier_uniform_(layer.lin.weight)
                if layer.lin.bias is not None:
                    nn.init.zeros_(layer.lin.bias)

class MessagePassing(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MessagePassing, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
    
    # ... existing code ... 