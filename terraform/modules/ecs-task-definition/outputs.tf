variable "region" {
  description = "The AWS region to deploy the ECS service"
  type        = string
}

output "ecs_task_definition_arn" {
  value = aws_ecs_task_definition.ecs_task_definition.arn
}