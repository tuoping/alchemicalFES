dilation=2
```
        self.convs = nn.ModuleList([layer for layer in [nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=args.kernel_size, padding=args.padding),
                      nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=args.kernel_size, padding=args.padding),
                      nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=args.kernel_size, dilation=2, padding=args.padding*2),
                      nn.Conv2d(args.hidden_dim, args.hidden_dim, kernel_size=args.kernel_size, dilation=2, padding=args.padding*2)
                      ] for i in range(args.num_cnn_stacks)])
```
