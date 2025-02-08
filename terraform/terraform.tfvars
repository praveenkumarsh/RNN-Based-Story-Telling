region            = "us-east-1"
vpc_cidr          = "10.0.0.0/16"
vpc_name          = "StoryGenerationVPC"
availability_zones = ["us-east-1a", "us-east-1b"]

allowed_ip        = "203.0.113.0/24"
cluster_name      = "ProjectCluster"
task_family       = "Story"
cpu               = "256"
memory            = "512"
container_name    = "StoryContainer"
image             = "public.ecr.aws/ecs-sample-image/amazon-ecs-sample:latest"
container_port    = 80
service_name      = "StoryService"
desired_count     = 1