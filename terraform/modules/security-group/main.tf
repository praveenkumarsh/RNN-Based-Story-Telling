// Define the security group for ECS
resource "aws_security_group" "ecs_security_group" {
  name        = "ecs_security_group"
  description = "Allow all egress and inbound access from specified IP"
  vpc_id      = var.vpc_id

  // Ingress rules to allow inbound access from the specified IP
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ip]
  }

  // Egress rules to allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}