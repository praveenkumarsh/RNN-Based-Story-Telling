output "service_id" {
  description = "The ID of the ECS service"
  value       = aws_ecs_service.this.id
}

# output "service_arn" {
#   description = "The ARN of the ECS service"
#   value       = aws_ecs_service.this.arn
# }