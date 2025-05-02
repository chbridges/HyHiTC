mv ./HIE/layers ./layers
mv ./HIE/manifolds ./manifolds
mv ./HIE/models ./models
mv ./HIE/utils ./utils
sed -i 's/torch.spmm(adj, x_tangent)/torch.matmul(adj, x_tangent)/' ./layers/hyp_layers.py
