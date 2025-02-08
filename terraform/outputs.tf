// Outputs 
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "vpc_name" {
  value = module.vpc.vpc_name
}

output "security_group_id" {
  value = module.security_group.security_group_id
}

resource "aws_ecs_cluster" "main" {
  name = var.cluster_name
}


output "cluster_arn" {
  value = aws_ecs_cluster.main.arn
}

output "cluster_id" {
  value = aws_ecs_cluster.main.id
}

output "cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_task_definition_arn" {
  value = module.ecs_task_definition.ecs_task_definition_arn
}
